"""Rollout buffer with GAE support for MAPPO."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class MAPPORolloutBuffer:
    """Store rollout tensors as [time, agent, ...] arrays."""

    def __init__(self, rollout_length: int, num_agents: int, obs_dim: int, state_dim: int, action_dim: int):
        self.rollout_length = rollout_length
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        self.obs: List[np.ndarray] = []
        self.global_states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.action_masks: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[float] = []
        self.values: List[float] = []
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def store(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        value: float,
    ) -> None:
        self.obs.append(obs.astype(np.float32))
        self.global_states.append(global_state.astype(np.float32))
        self.actions.append(actions.astype(np.int64))
        self.action_masks.append(action_masks.astype(np.float32))
        self.log_probs.append(log_probs.astype(np.float32))
        self.rewards.append(rewards.astype(np.float32))
        self.dones.append(float(done))
        self.values.append(float(value))

    @property
    def size(self) -> int:
        return len(self.rewards)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [float(last_value)], dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        for step in reversed(range(self.size)):
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * values[step + 1] * next_non_terminal - values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
        self.advantages = advantages
        self.returns = advantages + np.asarray(self.values, dtype=np.float32)[:, None]

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_returns_and_advantages before as_tensors().")
        obs = np.asarray(self.obs, dtype=np.float32)
        states = np.asarray(self.global_states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int64)
        masks = np.asarray(self.action_masks, dtype=np.float32)
        old_log_probs = np.asarray(self.log_probs, dtype=np.float32)
        advantages = self.advantages.astype(np.float32)
        returns = self.returns.astype(np.float32)
        old_values = np.asarray(self.values, dtype=np.float32)

        flat_agent_ids = np.tile(np.arange(self.num_agents), self.size)
        agent_onehot = np.eye(self.num_agents, dtype=np.float32)[flat_agent_ids]
        flat_states = np.repeat(states, self.num_agents, axis=0)
        flat_obs = obs.reshape(self.size * self.num_agents, self.obs_dim)
        flat_actions = actions.reshape(self.size * self.num_agents)
        flat_masks = masks.reshape(self.size * self.num_agents, self.action_dim)
        flat_log_probs = old_log_probs.reshape(self.size * self.num_agents)
        flat_advantages = advantages.reshape(self.size * self.num_agents)
        flat_critic_returns = np.repeat(returns.mean(axis=1), self.num_agents)
        flat_old_values = np.repeat(old_values, self.num_agents)
        norm_adv = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        return {
            "obs": torch.as_tensor(flat_obs, device=device),
            "agent_onehot": torch.as_tensor(agent_onehot, device=device),
            "actions": torch.as_tensor(flat_actions, device=device),
            "action_masks": torch.as_tensor(flat_masks, device=device),
            "old_log_probs": torch.as_tensor(flat_log_probs, device=device),
            "advantages": torch.as_tensor(norm_adv, device=device),
            "states": torch.as_tensor(flat_states, device=device),
            "critic_returns": torch.as_tensor(flat_critic_returns, device=device),
            "old_values": torch.as_tensor(flat_old_values, device=device),
        }
