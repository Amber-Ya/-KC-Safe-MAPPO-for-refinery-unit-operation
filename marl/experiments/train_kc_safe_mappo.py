"""Minimal training entry point for KC-Safe-MAPPO."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from types import ModuleType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from marl.algorithms.kc_safe_mappo import KCSafeMAPPOTrainer
from marl.configs.algo_config import ALGO_CONFIG
from marl.envs.refinery_env import RefinerySchedulingEnv
from marl.utils.config_adapter import ConfigAdapter


def load_config_module(path: str) -> ModuleType:
    config_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("refinery_case_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KC-Safe-MAPPO on the refinery scheduling case.")
    parser.add_argument("--config", default="config.py", help="Path to the existing case-data config.py.")
    parser.add_argument("--total_steps", type=int, default=100000, help="Number of environment steps to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", default="results/marl", help="Directory for MARL logs and checkpoints.")
    parser.add_argument("--rollout_length", type=int, default=None, help="Override rollout length.")
    parser.add_argument("--ppo_epochs", type=int, default=None, help="Override PPO epochs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_module = load_config_module(args.config)
    env_config = ConfigAdapter(config_module).build_env_config()
    env = RefinerySchedulingEnv(env_config, seed=args.seed)
    algo_config = dict(ALGO_CONFIG)
    if args.rollout_length is not None:
        algo_config["rollout_length"] = args.rollout_length
    if args.ppo_epochs is not None:
        algo_config["ppo_epochs"] = args.ppo_epochs
    trainer = KCSafeMAPPOTrainer(env, algo_config=algo_config, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logs = trainer.train(total_steps=args.total_steps, output_dir=args.output_dir)
    if logs:
        last = logs[-1]
        print(
            "KC-Safe-MAPPO training finished: "
            f"steps={last['env_steps']}, reward={last['total_reward']:.4f}, "
            f"profit={last.get('profit', 0.0):.2f}, cost={last['total_cost']:.4f}"
        )
    print(f"Logs and checkpoint written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
