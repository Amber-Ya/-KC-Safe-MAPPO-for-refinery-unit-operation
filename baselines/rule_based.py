"""Knowledge-guided rule-based scheduling baseline."""

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


def build_policy() -> PolicyFn:
    def policy(env: RefinerySchedulingEnv, obs: Mapping[str, np.ndarray]) -> Dict[str, int]:
        product_price = average_product_price_multiplier(env)
        crude_price = float(env.state.get("crude_price_multiplier", 1.0))
        cdu_target = 5 if crude_price / max(product_price, 1e-8) > 1.0 else 6
        actions = {agent: 6 for agent in env.agents}
        for cdu in ("CDU1", "CDU2"):
            if cdu in actions:
                actions[cdu] = cdu_target
        return actions

    return policy


def average_product_price_multiplier(env: RefinerySchedulingEnv) -> float:
    multipliers = list(env.state.get("price_multipliers", {}).values())
    return float(sum(multipliers) / len(multipliers)) if multipliers else 1.0


def run_baseline() -> PolicyFn:
    return build_policy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the rule-based refinery scheduling baseline.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--uncertainty_profile", default="moderate", choices=("none", "moderate", "stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=24)
    parser.add_argument("--output_dir", default="results/baselines/rule_based")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_config = build_env_config(args.config, args.uncertainty_profile)
    eval_seeds = [args.seed + 1000 + i for i in range(args.eval_episodes)]
    rows, methods = evaluate_methods(env_config, {"rule_based": run_baseline()}, eval_seeds)
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "comparison_returns.csv"), rows)
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"methods": methods}, fh, ensure_ascii=False, indent=2)
    print(json.dumps(methods, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

