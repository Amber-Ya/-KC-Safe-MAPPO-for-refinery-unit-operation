"""Rolling-horizon MPC-style optimization baseline.

This module provides a lightweight rolling lookahead baseline over the MARL
environment dynamics. It is intentionally named separately from the full
Gurobi MILP because the true stochastic/robust MILP remains a larger model
extension; this baseline gives the paper pipeline an online optimizer with
measured per-period decision time.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, Mapping

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common import PolicyFn, build_env_config, evaluate_methods, write_csv
from baselines.rule_based import build_policy as build_rule_policy
from marl.envs.refinery_env import RefinerySchedulingEnv


def build_policy(
    rng: np.random.Generator,
    candidates: int = 12,
    lookahead: int = 4,
) -> PolicyFn:
    fallback_policy = build_rule_policy()

    def policy(env: RefinerySchedulingEnv, obs: Mapping[str, np.ndarray]) -> Dict[str, int]:
        candidate_actions = _candidate_actions(env, rng, candidates)
        best_actions = candidate_actions[0]
        best_profit = -float("inf")
        for actions in candidate_actions:
            trial_env = copy.deepcopy(env)
            trial_obs = {agent: np.asarray(value).copy() for agent, value in obs.items()}
            profit = _simulate_candidate(trial_env, trial_obs, actions, fallback_policy, lookahead)
            if profit > best_profit:
                best_profit = profit
                best_actions = actions
        return best_actions

    return policy


def _candidate_actions(
    env: RefinerySchedulingEnv,
    rng: np.random.Generator,
    candidates: int,
) -> list[Dict[str, int]]:
    actions = []
    for level in (0, 3, 4, 5, 6):
        actions.append({agent: min(level, env.action_dim - 1) for agent in env.agents})
    while len(actions) < candidates:
        sampled = {}
        for agent in env.agents:
            feasible = np.flatnonzero(env.get_action_mask(agent) > 0.0)
            sampled[agent] = int(rng.choice(feasible)) if feasible.size else 0
        actions.append(sampled)
    return actions[:candidates]


def _simulate_candidate(
    env: RefinerySchedulingEnv,
    obs: Mapping[str, np.ndarray],
    first_actions: Mapping[str, int],
    fallback_policy: PolicyFn,
    lookahead: int,
) -> float:
    total_profit = 0.0
    trial_obs = dict(obs)
    for step in range(max(1, lookahead)):
        actions = dict(first_actions) if step == 0 else fallback_policy(env, trial_obs)
        trial_obs, _, dones, info = env.step(actions)
        total_profit += float(info.get("profit", 0.0))
        if bool(dones["__all__"]):
            break
    return total_profit


def run_baseline(seed: int, candidates: int, lookahead: int) -> PolicyFn:
    return build_policy(np.random.default_rng(seed), candidates=candidates, lookahead=lookahead)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the rolling-horizon MPC-style baseline.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--uncertainty_profile", default="moderate", choices=("none", "moderate", "stress"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=24)
    parser.add_argument("--rolling_candidates", type=int, default=12)
    parser.add_argument("--rolling_lookahead", type=int, default=4)
    parser.add_argument("--output_dir", default="results/baselines/rolling_mpc_milp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_config = build_env_config(args.config, args.uncertainty_profile)
    policy = run_baseline(args.seed, args.rolling_candidates, args.rolling_lookahead)
    rows, methods = evaluate_methods(
        env_config,
        {"Rolling MPC-MILP": policy},
        [args.seed + 1000 + i for i in range(args.eval_episodes)],
    )
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "comparison_returns.csv"), rows)
    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"methods": methods}, fh, ensure_ascii=False, indent=2)
    print(json.dumps(methods, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

