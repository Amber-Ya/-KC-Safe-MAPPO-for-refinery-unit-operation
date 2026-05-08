"""RL/MARL same-family baselines and ablations for KC-Safe-MAPPO."""

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

from baselines.common import PolicyFn, build_env_config, evaluate_methods, write_csv
from baselines.rule_based import build_policy as build_rule_policy
from marl.algorithms.kc_safe_mappo import KCSafeMAPPOTrainer
from marl.configs.algo_config import ALGO_CONFIG, REWARD_CONFIG
from marl.envs.refinery_env import RefinerySchedulingEnv


VARIANT_NOTES = {
    "mappo": "Shared-actor MAPPO without profit guard; safety repair remains in the environment.",
    "safe_mappo": "MAPPO with action masks/safety repair, without profit guard.",
    "kc_safe_mappo_no_profit_guard": "Full KC-Safe-MAPPO training setup, evaluated without profit guard.",
    "kc_safe_mappo_profit_guard": "KC-Safe-MAPPO evaluated with the rule-based profit guard.",
    "alpha_07": "KC-Safe-MAPPO with global profit reward alpha=0.70.",
    "alpha_095": "KC-Safe-MAPPO with global profit reward alpha=0.95.",
    "alpha_10": "KC-Safe-MAPPO with global profit reward alpha=1.00.",
}


def train_variant(
    variant: str,
    env_config: Mapping[str, object],
    seed: int,
    train_steps: int,
    output_dir: str,
    entropy_coef: float,
    learning_rate: float,
) -> tuple[PolicyFn, dict[str, object]]:
    reward_config = dict(REWARD_CONFIG)
    if variant == "alpha_07":
        reward_config["alpha"] = 0.70
    elif variant == "alpha_10":
        reward_config["alpha"] = 1.00
    elif variant == "alpha_095":
        reward_config["alpha"] = 0.95

    algo_config = dict(ALGO_CONFIG)
    algo_config["entropy_coef"] = entropy_coef
    algo_config["learning_rate"] = learning_rate
    env = RefinerySchedulingEnv(dict(env_config), seed=seed, reward_config=reward_config)
    trainer = KCSafeMAPPOTrainer(env, algo_config=algo_config, seed=seed, device="cpu")
    variant_dir = os.path.join(output_dir, variant)
    logs = trainer.train(total_steps=train_steps, output_dir=variant_dir)
    if trainer.best_checkpoint_path and os.path.exists(trainer.best_checkpoint_path):
        trainer.load_checkpoint(trainer.best_checkpoint_path)
    profit_guard = variant == "kc_safe_mappo_profit_guard"
    return make_mappo_policy(trainer, profit_guard=profit_guard), {
        "variant": variant,
        "note": VARIANT_NOTES.get(variant, ""),
        "train_steps": train_steps,
        "last_train_profit": logs[-1]["profit"] if logs else None,
        "best_train_profit": trainer.best_training_profit,
        "reward_alpha": reward_config.get("alpha"),
        "output_dir": variant_dir,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL/MARL same-family baselines and ablations.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--uncertainty_profile", default="moderate", choices=("moderate", "stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_steps", type=int, default=12000)
    parser.add_argument("--eval_episodes", type=int, default=24)
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["safe_mappo", "kc_safe_mappo_no_profit_guard", "kc_safe_mappo_profit_guard", "alpha_07", "alpha_095", "alpha_10"],
        choices=sorted(VARIANT_NOTES),
    )
    parser.add_argument("--output_dir", default="results/baselines/marl_variants")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    env_config = build_env_config(args.config, args.uncertainty_profile)
    policies: Dict[str, PolicyFn] = {}
    training_summary: Dict[str, object] = {}
    for idx, variant in enumerate(args.variants):
        policy, metadata = train_variant(
            variant=variant,
            env_config=env_config,
            seed=args.seed + idx,
            train_steps=args.train_steps,
            output_dir=args.output_dir,
            entropy_coef=args.entropy_coef,
            learning_rate=args.learning_rate,
        )
        policies[variant] = policy
        training_summary[variant] = metadata
    rows, methods = evaluate_methods(
        env_config,
        policies,
        [args.seed + 1000 + i for i in range(args.eval_episodes)],
    )
    write_csv(os.path.join(args.output_dir, "comparison_returns.csv"), rows)
    summary = {
        "uncertainty_profile": args.uncertainty_profile,
        "training_summary": training_summary,
        "methods": methods,
    }
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(methods, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

