"""Genetic algorithm baseline for discrete refinery load scheduling."""

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
    population: int,
    generations: int,
    blocks: int,
    rng: np.random.Generator,
) -> tuple[PolicyFn, dict[str, float]]:
    agent_list = list(agents)
    spec = PlanSearchSpec(dict(env_config), agent_list, int(action_dim), int(horizon), int(blocks), scenario_seeds)
    population_plans = _initial_population(population, spec.blocks, len(agent_list), spec.action_dim, rng)
    best_plan = population_plans[0]
    best_scores = plan_scores(spec, best_plan)
    best_objective = float(mean(best_scores))

    for _ in range(max(1, generations)):
        scored = []
        for plan in population_plans:
            scores = plan_scores(spec, plan)
            objective = float(mean(scores))
            scored.append((objective, plan, scores))
            if objective > best_objective:
                best_objective = objective
                best_plan = plan.copy()
                best_scores = scores
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = [item[1] for item in scored[: max(2, population // 4)]]
        next_population = [elite.copy() for elite in elites]
        while len(next_population) < population:
            parent_a, parent_b = rng.choice(len(elites), size=2, replace=True)
            child = _crossover(elites[int(parent_a)], elites[int(parent_b)], rng)
            next_population.append(_mutate(child, spec.action_dim, rng))
        population_plans = next_population

    plan = expand_blocks(best_plan, spec.horizon)
    return make_table_policy(plan, agent_list), summarize_plan_scores(best_scores, best_objective)


def _initial_population(
    population: int,
    blocks: int,
    num_agents: int,
    action_dim: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    plans = list(seeded_block_plans(blocks, num_agents))
    while len(plans) < population:
        plan = rng.integers(0, action_dim, size=(blocks, num_agents), endpoint=False)
        if rng.random() < 0.70:
            plan = np.maximum(plan, 3)
        plans.append(plan.astype(np.int64))
    return plans[:population]


def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(parent_a.shape) < 0.5
    return np.where(mask, parent_a, parent_b).astype(np.int64)


def _mutate(plan: np.ndarray, action_dim: int, rng: np.random.Generator, rate: float = 0.08) -> np.ndarray:
    child = plan.copy()
    mask = rng.random(child.shape) < rate
    child[mask] = rng.integers(0, action_dim, size=int(mask.sum()), endpoint=False)
    return child


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the genetic algorithm scheduling baseline.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--uncertainty_profile", default="moderate", choices=("none", "moderate", "stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=24)
    parser.add_argument("--planner_scenarios", type=int, default=8)
    parser.add_argument("--planner_blocks", type=int, default=4)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--output_dir", default="results/baselines/genetic_algorithm")
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
        population=args.population,
        generations=args.generations,
        blocks=args.planner_blocks,
        rng=np.random.default_rng(args.seed),
    )
    rows, methods = evaluate_methods(
        env_config,
        {"Genetic Algorithm": policy},
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

