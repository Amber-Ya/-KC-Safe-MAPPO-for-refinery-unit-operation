"""Scenario average approximation baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Any, Dict, Iterable, Mapping

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common import (
    PolicyFn,
    build_env_config,
    evaluate_methods,
    evaluate_policy,
    make_table_policy,
    write_csv,
)
from marl.envs.refinery_env import RefinerySchedulingEnv


class StochasticSAAPlanner:
    """Scenario-based discrete planner optimizing mean training-scenario profit."""

    def __init__(
        self,
        env_config: Mapping[str, Any],
        agents: Iterable[str],
        action_dim: int,
        horizon: int,
        blocks: int,
        rng: np.random.Generator,
    ):
        self.env_config = dict(env_config)
        self.agents = list(agents)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.blocks = max(1, int(blocks))
        self.rng = rng

    def optimize(self, scenario_seeds: list[int], candidates: int) -> tuple[np.ndarray, Dict[str, float]]:
        best_plan = self._seed_plan(4)
        best_scores = self._evaluate_plan(best_plan, scenario_seeds)
        best_objective = float(mean(best_scores))
        for plan in self._candidate_plans(candidates):
            scores = self._evaluate_plan(plan, scenario_seeds)
            value = float(mean(scores))
            if value > best_objective:
                best_plan = plan
                best_scores = scores
                best_objective = value
        return best_plan, {
            "objective": best_objective,
            "mean_profit": float(mean(best_scores)),
            "min_profit": float(min(best_scores)),
            "max_profit": float(max(best_scores)),
        }

    def _candidate_plans(self, candidates: int) -> Iterable[np.ndarray]:
        for level in (0, 3, 4, 5, 6):
            yield self._seed_plan(level)
        for _ in range(max(0, candidates - 5)):
            block_actions = self.rng.integers(
                0,
                self.action_dim,
                size=(self.blocks, len(self.agents)),
                endpoint=False,
            )
            if self.rng.random() < 0.65:
                block_actions = np.maximum(block_actions, 3)
            yield self._expand_blocks(block_actions)

    def _seed_plan(self, level: int) -> np.ndarray:
        return np.full((self.horizon, len(self.agents)), int(level), dtype=np.int64)

    def _expand_blocks(self, block_actions: np.ndarray) -> np.ndarray:
        plan = np.zeros((self.horizon, len(self.agents)), dtype=np.int64)
        for t in range(self.horizon):
            block = min(self.blocks - 1, int(t * self.blocks / self.horizon))
            plan[t] = block_actions[block]
        return plan

    def _evaluate_plan(self, plan: np.ndarray, scenario_seeds: list[int]) -> list[float]:
        policy = make_table_policy(plan, self.agents)
        return [evaluate_policy(self.env_config, policy, seed)["profit"] for seed in scenario_seeds]


def run_baseline(
    env_config: Mapping[str, Any],
    agents: Iterable[str],
    action_dim: int,
    horizon: int,
    scenario_seeds: list[int],
    candidates: int,
    blocks: int,
    rng: np.random.Generator,
) -> tuple[PolicyFn, Dict[str, float]]:
    planner = StochasticSAAPlanner(env_config, agents, action_dim, horizon, blocks, rng)
    plan, train_scores = planner.optimize(scenario_seeds, candidates)
    return make_table_policy(plan, agents), train_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the stochastic programming SAA baseline.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--uncertainty_profile", default="moderate", choices=("none", "moderate", "stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=24)
    parser.add_argument("--planner_candidates", type=int, default=80)
    parser.add_argument("--planner_scenarios", type=int, default=8)
    parser.add_argument("--planner_blocks", type=int, default=4)
    parser.add_argument("--output_dir", default="results/baselines/stochastic_programming_saa")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_config = build_env_config(args.config, args.uncertainty_profile)
    env = RefinerySchedulingEnv(env_config, seed=args.seed)
    policy, train_scores = run_baseline(
        env_config=env_config,
        agents=env.agents,
        action_dim=env.action_dim,
        horizon=env.horizon,
        scenario_seeds=[args.seed + 100 + i for i in range(args.planner_scenarios)],
        candidates=args.planner_candidates,
        blocks=args.planner_blocks,
        rng=np.random.default_rng(args.seed),
    )
    eval_seeds = [args.seed + 1000 + i for i in range(args.eval_episodes)]
    rows, methods = evaluate_methods(env_config, {"stochastic_programming_saa": policy}, eval_seeds)
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "comparison_returns.csv"), rows)
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"planner_train_scores": train_scores, "methods": methods}, fh, ensure_ascii=False, indent=2)
    print(json.dumps(methods, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

