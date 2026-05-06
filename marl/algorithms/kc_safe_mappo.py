"""KC-Safe-MAPPO trainer for refinery multi-unit scheduling."""

from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn

from marl.algorithms.networks import CentralizedCritic, SharedActor
from marl.algorithms.rollout_buffer import MAPPORolloutBuffer
from marl.configs.algo_config import ALGO_CONFIG


class KCSafeMAPPOTrainer:
    """Centralized critic + decentralized shared actor with action masks."""

    def __init__(
        self,
        env: Any,
        algo_config: Mapping[str, Any] | None = None,
        seed: int = 42,
        device: str | None = None,
    ):
        self.env = env
        self.config = dict(ALGO_CONFIG)
        if algo_config:
            self.config.update(dict(algo_config))
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        obs = self.env.reset(seed=seed)
        self.agents = list(self.env.agents)
        self.num_agents = len(self.agents)
        self.obs_dim = len(next(iter(obs.values())))
        self.state_dim = len(self.env.get_global_state())
        self.action_dim = env.action_dim

        hidden_dim = int(self.config["hidden_dim"])
        self.actor = SharedActor(self.obs_dim, self.action_dim, self.num_agents, hidden_dim).to(self.device)
        self.critic = CentralizedCritic(self.state_dim, hidden_dim).to(self.device)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(self.config["learning_rate"]))
        self._current_entropy_coef = float(self.config["entropy_coef"])
        self.last_losses: Dict[str, float] = {}

    def select_actions(self, obs: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        actions: Dict[str, int] = {}
        log_probs: Dict[str, float] = {}
        masks: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for idx, agent in enumerate(self.agents):
                obs_tensor = torch.as_tensor(obs[agent], dtype=torch.float32, device=self.device).unsqueeze(0)
                onehot = torch.eye(self.num_agents, device=self.device)[idx].unsqueeze(0)
                mask_np = self.env.get_action_mask(agent)
                mask = torch.as_tensor(mask_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, log_prob, _ = self.actor.sample(obs_tensor, onehot, mask)
                actions[agent] = int(action.item())
                log_probs[agent] = float(log_prob.item())
                masks[agent] = mask_np
        return {"actions": actions, "log_probs": log_probs, "masks": masks}

    def collect_rollout(self, obs: Mapping[str, np.ndarray], rollout_length: int) -> tuple[MAPPORolloutBuffer, Dict[str, np.ndarray], Dict[str, float]]:
        buffer = MAPPORolloutBuffer(rollout_length, self.num_agents, self.obs_dim, self.state_dim, self.action_dim)
        episode_metrics: Dict[str, float] = {
            "episode_reward": 0.0,
            "revenue": 0.0,
            "profit": 0.0,
            "total_cost": 0.0,
            "energy_cost": 0.0,
            "switch_cost": 0.0,
            "shortage_cost": 0.0,
            "inventory_penalty": 0.0,
            "violation_penalty": 0.0,
            "demand_satisfaction_rate": 0.0,
            "inventory_violation_count": 0.0,
            "unit_switch_count": 0.0,
        }
        metric_steps = 0

        for _ in range(rollout_length):
            global_state = self.env.get_global_state()
            with torch.no_grad():
                value = self.critic(
                    torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                ).item()
            selected = self.select_actions(obs)
            actions = selected["actions"]
            next_obs, rewards, dones, infos = self.env.step(actions)

            obs_array = np.stack([obs[agent] for agent in self.agents])
            action_array = np.asarray([actions[agent] for agent in self.agents], dtype=np.int64)
            mask_array = np.stack([selected["masks"][agent] for agent in self.agents])
            log_prob_array = np.asarray([selected["log_probs"][agent] for agent in self.agents], dtype=np.float32)
            reward_array = np.asarray([rewards[agent] for agent in self.agents], dtype=np.float32)
            buffer.store(obs_array, global_state, action_array, mask_array, log_prob_array, reward_array, dones["__all__"], value)

            episode_metrics["episode_reward"] += float(np.mean(reward_array))
            for key in [
                "revenue",
                "profit",
                "total_cost",
                "energy_cost",
                "switch_cost",
                "shortage_cost",
                "inventory_penalty",
                "violation_penalty",
                "inventory_violation_count",
                "unit_switch_count",
            ]:
                episode_metrics[key] += float(infos.get(key, 0.0))
            episode_metrics["demand_satisfaction_rate"] += float(infos.get("demand_satisfaction_rate", 0.0))
            metric_steps += 1

            obs = next_obs
            if dones["__all__"]:
                obs = self.env.reset()

        if metric_steps:
            episode_metrics["demand_satisfaction_rate"] /= metric_steps
        return buffer, dict(obs), episode_metrics

    def update(self, buffer: MAPPORolloutBuffer) -> Dict[str, float]:
        with torch.no_grad():
            last_value = self.critic(
                torch.as_tensor(self.env.get_global_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
            ).item()
        buffer.compute_returns_and_advantages(
            last_value,
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        tensors = buffer.as_tensors(self.device)
        total = tensors["actions"].shape[0]
        batch_size = min(int(self.config["mini_batch_size"]), total)
        losses = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "loss": 0.0}
        updates = 0

        for _ in range(int(self.config["ppo_epochs"])):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, batch_size):
                idx = permutation[start : start + batch_size]
                dist = self.actor.distribution(
                    tensors["obs"][idx],
                    tensors["agent_onehot"][idx],
                    tensors["action_masks"][idx],
                )
                new_log_probs = dist.log_prob(tensors["actions"][idx])
                ratio = torch.exp(new_log_probs - tensors["old_log_probs"][idx])
                adv = tensors["advantages"][idx]
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(self.config["clip_param"]),
                    1.0 + float(self.config["clip_param"]),
                ) * adv
                actor_loss = -torch.min(ratio * adv, clipped).mean()
                entropy = dist.entropy().mean()

                values = self.critic(tensors["states"][idx])
                old_values = tensors["old_values"][idx]
                returns = tensors["critic_returns"][idx]
                value_clip = float(self.config.get("value_clip_param", self.config["clip_param"]))
                values_clipped = old_values + torch.clamp(
                    values - old_values,
                    -value_clip,
                    value_clip,
                )
                value_loss = (values - returns).pow(2)
                value_loss_clipped = (values_clipped - returns).pow(2)
                critic_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()
                loss = (
                    actor_loss
                    + float(self.config["value_loss_coef"]) * critic_loss
                    - self._current_entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    float(self.config["max_grad_norm"]),
                )
                self.optimizer.step()
                losses["actor_loss"] += float(actor_loss.item())
                losses["critic_loss"] += float(critic_loss.item())
                losses["entropy"] += float(entropy.item())
                losses["loss"] += float(loss.item())
                updates += 1

        self.last_losses = {key: value / max(1, updates) for key, value in losses.items()}
        return self.last_losses

    def train(self, total_steps: int, output_dir: str = "results/marl") -> list[Dict[str, float]]:
        os.makedirs(output_dir, exist_ok=True)
        obs = self.env.reset(seed=self.seed)
        rollout_length = int(self.config["rollout_length"])
        logs: list[Dict[str, float]] = []
        env_steps = 0
        episode = 0
        entropy_start = float(self.config["entropy_coef"])
        entropy_end = entropy_start * 0.1  # anneal to 10% of initial
        lr_start = float(self.config["learning_rate"])
        lr_end = lr_start * 0.1
        self._current_entropy_coef = entropy_start
        while env_steps < total_steps:
            # --- Linear annealing ---
            progress = min(1.0, env_steps / max(1, total_steps))
            self._current_entropy_coef = entropy_start + (entropy_end - entropy_start) * progress
            current_lr = lr_start + (lr_end - lr_start) * progress
            for pg in self.optimizer.param_groups:
                pg["lr"] = current_lr

            current_rollout = min(rollout_length, total_steps - env_steps)
            buffer, obs, metrics = self.collect_rollout(obs, current_rollout)
            losses = self.update(buffer)
            env_steps += current_rollout
            episode += 1
            row = {
                "episode": episode,
                "env_steps": env_steps,
                "total_reward": metrics["episode_reward"],
                "revenue": metrics["revenue"],
                "profit": metrics["profit"],
                "total_cost": metrics["total_cost"],
                "energy_cost": metrics["energy_cost"],
                "switch_cost": metrics["switch_cost"],
                "shortage_cost": metrics["shortage_cost"],
                "inventory_penalty": metrics["inventory_penalty"],
                "violation_penalty": metrics["violation_penalty"],
                "demand_satisfaction_rate": metrics["demand_satisfaction_rate"],
                "inventory_violation_count": int(metrics["inventory_violation_count"]),
                "unit_switch_count": int(metrics["unit_switch_count"]),
                **losses,
            }
            logs.append(row)
            self._write_logs(os.path.join(output_dir, "training_log.csv"), logs)
        self.save_checkpoint(os.path.join(output_dir, "kc_safe_mappo_checkpoint.pt"))
        return logs

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "num_agents": self.num_agents,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config.update(checkpoint.get("config", {}))

    def _write_logs(self, path: str, logs: list[Dict[str, float]]) -> None:
        if not logs:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(logs[-1].keys()))
            writer.writeheader()
            writer.writerows(logs)
