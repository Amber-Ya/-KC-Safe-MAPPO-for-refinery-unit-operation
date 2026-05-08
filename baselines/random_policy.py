"""Random feasible-action baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Mapping

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common import PolicyFn, build_env_config, evaluate_methods, write_csv
from marl.envs.refinery_env import RefinerySchedulingEnv


def build_policy(rng: np.random.Generator) -> PolicyFn:
    def policy(env: RefinerySchedulingEnv, obs: Mapping[str, np.ndarray]) -> Dict[str, int]:
        actions = {}
        for agent in env.agents:
            feasible = np.flatnonzero(env.get_action_mask(agent) > 0.0)
            actions[agent] = int(rng.choice(feasible)) if feasible.size else 0
        return actions

    return policy


def run_baseline(seed: int) -> PolicyFn:
    return build_policy(np.random.default_rng(seed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the random feasible-action baseline.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--uncertainty_profile", default="moderate", choices=("none", "moderate", "stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=24)
    parser.add_argument("--output_dir", default="results/baselines/random_policy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_config = build_env_config(args.config, args.uncertainty_profile)
    eval_seeds = [args.seed + 1000 + i for i in range(args.eval_episodes)]
    rows, methods = evaluate_methods(
        env_config,
        {"random_policy": run_baseline(args.seed + 999)},
        eval_seeds,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "comparison_returns.csv"), rows)
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"methods": methods}, fh, ensure_ascii=False, indent=2)
    print(json.dumps(methods, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

