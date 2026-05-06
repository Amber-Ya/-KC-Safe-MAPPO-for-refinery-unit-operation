"""Neural networks for centralized-critic decentralized-actor MAPPO."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class SharedActor(nn.Module):
    """Shared decentralized actor conditioned on one-hot agent identity."""

    def __init__(self, obs_dim: int, action_dim: int, agent_id_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + agent_id_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        agent_id_onehot: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.net(torch.cat([obs, agent_id_onehot], dim=-1))
        if action_mask is not None:
            logits = logits + torch.log(action_mask.clamp_min(1e-8))
        return logits

    def distribution(
        self,
        obs: torch.Tensor,
        agent_id_onehot: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> Categorical:
        return Categorical(logits=self.forward(obs, agent_id_onehot, action_mask))

    def sample(
        self,
        obs: torch.Tensor,
        agent_id_onehot: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs, agent_id_onehot, action_mask)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class CentralizedCritic(nn.Module):
    """Centralized value function over the flattened global state."""

    def __init__(self, global_state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)
