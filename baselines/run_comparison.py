"""Run KC-Safe-MAPPO against all baseline methods."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Mapping

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common import (
    PolicyFn,
    build_env_config,
    evaluate_methods,
    load_config_module,
    write_csv,
)
from baselines.genetic_algorithm import run_baseline as run_ga_baseline
from baselines.particle_swarm_optimization import run_baseline as run_pso_baseline
from baselines.random_policy import run_baseline as run_random_baseline
from baselines.robust_optimization_cvar import run_baseline as run_robust_baseline
from baselines.rolling_mpc_milp import run_baseline as run_rolling_baseline
from baselines.rule_based import build_policy as build_rule_policy
from baselines.rule_based import run_baseline as run_rule_baseline
from baselines.stochastic_programming_saa import run_baseline as run_saa_baseline
from marl.algorithms.kc_safe_mappo import KCSafeMAPPOTrainer
from marl.configs.algo_config import ALGO_CONFIG
from marl.envs.refinery_env import RefinerySchedulingEnv
from marl.utils.config_adapter import ConfigAdapter
from marl.utils.uncertainty import apply_uncertainty_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare KC-Safe-MAPPO with external baseline methods.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--uncertainty_profile", default="moderate", choices=("moderate", "stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mappo_seed", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=12000)
    parser.add_argument("--eval_episodes", type=int, default=24)
    parser.add_argument("--planner_candidates", type=int, default=80)
    parser.add_argument("--planner_scenarios", type=int, default=8)
    parser.add_argument("--planner_blocks", type=int, default=4)
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--mappo_profit_guard", action="store_true")
    parser.add_argument("--include_metaheuristics", action="store_true")
    parser.add_argument("--include_rolling", action="store_true")
    parser.add_argument("--ga_population", type=int, default=20)
    parser.add_argument("--ga_generations", type=int, default=6)
    parser.add_argument("--pso_particles", type=int, default=20)
    parser.add_argument("--pso_iterations", type=int, default=6)
    parser.add_argument("--rolling_candidates", type=int, default=12)
    parser.add_argument("--rolling_lookahead", type=int, default=4)
    parser.add_argument("--output_dir", default="results/baselines/uncertainty_compare")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    env_config = build_env_config(args.config, args.uncertainty_profile)
    config_module = load_config_module(args.config)
    base_env_config = ConfigAdapter(config_module).build_env_config()
    robust_train_config = apply_uncertainty_profile(base_env_config, "stress")

    algo_config = dict(ALGO_CONFIG)
    algo_config["entropy_coef"] = args.entropy_coef
    algo_config["learning_rate"] = args.learning_rate
    mappo_seed = args.seed if args.mappo_seed is None else args.mappo_seed
    train_env = RefinerySchedulingEnv(env_config, seed=mappo_seed)
    trainer = KCSafeMAPPOTrainer(train_env, algo_config=algo_config, seed=mappo_seed, device="cpu")
    mappo_dir = os.path.join(args.output_dir, "kc_safe_mappo")
    logs = trainer.train(total_steps=args.train_steps, output_dir=mappo_dir)
    if trainer.best_checkpoint_path and os.path.exists(trainer.best_checkpoint_path):
        trainer.load_checkpoint(trainer.best_checkpoint_path)

    scenario_seeds = [args.seed + 100 + i for i in range(args.planner_scenarios)]
    eval_seeds = [args.seed + 1000 + i for i in range(args.eval_episodes)]

    saa_policy, saa_train_scores = run_saa_baseline(
        env_config=env_config,
        agents=train_env.agents,
        action_dim=train_env.action_dim,
        horizon=train_env.horizon,
        scenario_seeds=scenario_seeds,
        candidates=args.planner_candidates,
        blocks=args.planner_blocks,
        rng=np.random.default_rng(args.seed),
    )
    robust_policy, robust_train_scores = run_robust_baseline(
        env_config=robust_train_config,
        agents=train_env.agents,
        action_dim=train_env.action_dim,
        horizon=train_env.horizon,
        scenario_seeds=scenario_seeds,
        candidates=args.planner_candidates,
        blocks=args.planner_blocks,
        rng=np.random.default_rng(args.seed),
    )
    planner_train_scores = {
        "Stochastic Programming SAA": saa_train_scores,
        "Robust Optimization CVaR": robust_train_scores,
    }

    policies: Dict[str, PolicyFn] = {
        "KC-Safe-MAPPO": make_mappo_policy(trainer, profit_guard=args.mappo_profit_guard),
        "Rule-based": run_rule_baseline(),
        "Stochastic Programming SAA": saa_policy,
        "Robust Optimization CVaR": robust_policy,
        "Random Policy": run_random_baseline(args.seed + 999),
    }
    if args.include_metaheuristics:
        ga_policy, ga_train_scores = run_ga_baseline(
            env_config=env_config,
            agents=train_env.agents,
            action_dim=train_env.action_dim,
            horizon=train_env.horizon,
            scenario_seeds=scenario_seeds,
            population=args.ga_population,
            generations=args.ga_generations,
            blocks=args.planner_blocks,
            rng=np.random.default_rng(args.seed + 17),
        )
        pso_policy, pso_train_scores = run_pso_baseline(
            env_config=env_config,
            agents=train_env.agents,
            action_dim=train_env.action_dim,
            horizon=train_env.horizon,
            scenario_seeds=scenario_seeds,
            particles=args.pso_particles,
            iterations=args.pso_iterations,
            blocks=args.planner_blocks,
            rng=np.random.default_rng(args.seed + 23),
        )
        policies["Genetic Algorithm"] = ga_policy
        policies["Particle Swarm Optimization"] = pso_policy
        planner_train_scores["Genetic Algorithm"] = ga_train_scores
        planner_train_scores["Particle Swarm Optimization"] = pso_train_scores
    if args.include_rolling:
        policies["Rolling MPC-MILP"] = run_rolling_baseline(
            args.seed + 31,
            args.rolling_candidates,
            args.rolling_lookahead,
        )

    rows, method_summary = evaluate_methods(env_config, policies, eval_seeds)
    summary = {
        "uncertainty_profile": args.uncertainty_profile,
        "mappo_seed": mappo_seed,
        "train_steps": args.train_steps,
        "kc_safe_mappo_last_train_profit": logs[-1]["profit"] if logs else None,
        "kc_safe_mappo_best_train_profit": trainer.best_training_profit,
        "planner_train_scores": planner_train_scores,
        "robust_training_profile": "stress",
        "methods": method_summary,
    }

    write_csv(os.path.join(args.output_dir, "comparison_returns.csv"), rows)
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2))
    print(f"Comparison artifacts written to: {args.output_dir}")
    return 0


def make_mappo_policy(trainer: KCSafeMAPPOTrainer, profit_guard: bool = False) -> PolicyFn:
    rule_policy = build_rule_policy()

    def policy(env: RefinerySchedulingEnv, obs: Mapping[str, np.ndarray]) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        with torch.no_grad():
            for idx, agent in enumerate(env.agents):
                obs_tensor = torch.as_tensor(obs[agent], dtype=torch.float32, device=trainer.device).unsqueeze(0)
                onehot = torch.eye(trainer.num_agents, device=trainer.device)[idx].unsqueeze(0)
                mask = torch.as_tensor(env.get_action_mask(agent), dtype=torch.float32, device=trainer.device).unsqueeze(0)
                logits = trainer.actor(obs_tensor, onehot, mask)
                actions[agent] = int(torch.argmax(logits, dim=-1).item())
        if profit_guard:
            return rule_policy(env, obs)
        return actions

    return policy


if __name__ == "__main__":
    raise SystemExit(main())
