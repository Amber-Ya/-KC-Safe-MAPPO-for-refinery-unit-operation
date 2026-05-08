"""Shared plan-search helpers for non-RL scheduling baselines."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np

from baselines.common import evaluate_policy, make_table_policy


@dataclass
class PlanSearchSpec:
    env_config: Mapping[str, Any]
    agents: list[str]
    action_dim: int
    horizon: int
    blocks: int
    scenario_seeds: list[int]


def expand_blocks(block_actions: np.ndarray, horizon: int) -> np.ndarray:
    blocks = max(1, int(block_actions.shape[0]))
    plan = np.zeros((horizon, block_actions.shape[1]), dtype=np.int64)
    for t in range(horizon):
        block = min(blocks - 1, int(t * blocks / horizon))
        plan[t] = block_actions[block]
    return plan


def seed_block_plan(level: int, blocks: int, num_agents: int) -> np.ndarray:
    return np.full((max(1, blocks), num_agents), int(level), dtype=np.int64)


def plan_scores(spec: PlanSearchSpec, block_actions: np.ndarray) -> list[float]:
    plan = expand_blocks(block_actions, spec.horizon)
    policy = make_table_policy(plan, spec.agents)
    return [
        evaluate_policy(spec.env_config, policy, seed)["profit"]
        for seed in spec.scenario_seeds
    ]


def summarize_plan_scores(scores: list[float], objective: float) -> dict[str, float]:
    return {
        "objective": float(objective),
        "mean_profit": float(mean(scores)),
        "min_profit": float(min(scores)),
        "max_profit": float(max(scores)),
    }


def seeded_block_plans(blocks: int, num_agents: int) -> Iterable[np.ndarray]:
    for level in (0, 3, 4, 5, 6):
        yield seed_block_plan(level, blocks, num_agents)

