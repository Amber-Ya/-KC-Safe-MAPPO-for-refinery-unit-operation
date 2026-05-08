"""Discrete particle swarm optimization baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common import PolicyFn, build_env_config, evaluate_methods, make_table_policy, write_csv
from baselines.plan_search import PlanSearchSpec, expand_blocks, plan_scores, seeded_block_plans, summarize_plan_scores
from marl.envs.refinery_env import RefinerySchedulingEnv


def run_baseline(
    env_config: Mapping[str, Any],
    agents: Iterable[str],
    action_dim: int,
    horizon: int,
    scenario_seeds: list[int],
    particles: int,
    iterations: int,
    blocks: int,
    rng: np.random.Generator,
) -> tuple[PolicyFn, dict[str, float]]:
    agent_list = list(agents)
    spec = PlanSearchSpec(dict(env_config), agent_list, int(action_dim), int(horizon), int(blocks), scenario_seeds)
    shape = (spec.blocks, len(agent_list))
    positions = rng.uniform(0.0, action_dim - 1, size=(particles, *shape))
    for idx, seed_plan in enumerate(seeded_block_plans(spec.blocks, len(agent_list))):
        if idx < particles:
            positions[idx] = seed_plan
    velocities = rng.normal(0.0, 0.5, size=positions.shape)
    personal_best = positions.copy()
    personal_scores = np.full(particles, -np.inf, dtype=np.float64)
    global_best = positions[0].copy()
    global_scores = plan_scores(spec, _discretize(global_best, action_dim))
    global_objective = float(mean(global_scores))

    for _ in range(max(1, iterations)):
        for idx in range(particles):
            block_plan = _discretize(positions[idx], action_dim)
            scores = plan_scores(spec, block_plan)
            objective = float(mean(scores))
            if objective > personal_scores[idx]:
                personal_scores[idx] = objective
                personal_best[idx] = positions[idx].copy()
            if objective > global_objective:
                global_objective = objective
                global_best = positions[idx].copy()
                global_scores = scores
        inertia = 0.55
        cognitive = 1.25 * rng.random(positions.shape) * (personal_best - positions)
        social = 1.25 * rng.random(positions.shape) * (global_best - positions)
        velocities = inertia * velocities + cognitive + social
        positions = np.clip(positions + velocities, 0.0, action_dim - 1)

    best_block_plan = _discretize(global_best, action_dim)
    plan = expand_blocks(best_block_plan, spec.horizon)
    return make_table_policy(plan, agent_list), summarize_plan_scores(global_scores, global_objective)


def _discretize(position: np.ndarray, action_dim: int) -> np.ndarray:
    return np.rint(np.clip(position, 0.0, action_dim - 1)).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the particle swarm optimization baseline.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--uncertainty_profile", default="moderate", choices=("none", "moderate", "stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=24)
    parser.add_argument("--planner_scenarios", type=int, default=8)
    parser.add_argument("--planner_blocks", type=int, default=4)
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--output_dir", default="results/baselines/particle_swarm_optimization")
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
        particles=args.particles,
        iterations=args.iterations,
        blocks=args.planner_blocks,
        rng=np.random.default_rng(args.seed),
    )
    rows, methods = evaluate_methods(
        env_config,
        {"Particle Swarm Optimization": policy},
        [args.seed + 1000 + i for i in range(args.eval_episodes)],
    )
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "comparison_returns.csv"), rows)
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"planner_train_scores": train_scores, "methods": methods}, fh, ensure_ascii=False, indent=2)
    print(json.dumps(methods, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

