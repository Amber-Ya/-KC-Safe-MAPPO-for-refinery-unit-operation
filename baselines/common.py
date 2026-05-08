"""Shared utilities for refinery scheduling baseline experiments."""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
import time
from statistics import mean, pstdev
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Mapping

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from marl.envs.refinery_env import RefinerySchedulingEnv
from marl.utils.config_adapter import ConfigAdapter
from marl.utils.uncertainty import apply_uncertainty_profile

PolicyFn = Callable[[RefinerySchedulingEnv, Mapping[str, np.ndarray]], Dict[str, int]]


def load_config_module(path: str) -> ModuleType:
    config_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("refinery_case_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_env_config(config_path: str, uncertainty_profile: str) -> Dict[str, Any]:
    config_module = load_config_module(config_path)
    base_env_config = ConfigAdapter(config_module).build_env_config()
    return apply_uncertainty_profile(base_env_config, uncertainty_profile)


def make_table_policy(plan: np.ndarray, agents: Iterable[str]) -> PolicyFn:
    agent_list = list(agents)

    def policy(env: RefinerySchedulingEnv, obs: Mapping[str, np.ndarray]) -> Dict[str, int]:
        t = min(int(env.state.get("t", 0)), plan.shape[0] - 1)
        return {agent: int(plan[t, idx]) for idx, agent in enumerate(agent_list)}

    return policy


def evaluate_policy(
    env_config: Mapping[str, Any],
    policy: PolicyFn,
    seed: int,
) -> Dict[str, float]:
    env = RefinerySchedulingEnv(dict(env_config), seed=seed)
    obs = env.reset(seed=seed)
    totals = {
        "revenue": 0.0,
        "profit": 0.0,
        "total_cost": 0.0,
        "inventory_violation_count": 0.0,
        "unit_switch_count": 0.0,
        "demand_satisfaction_rate": 0.0,
    }
    decision_time = 0.0
    steps = 0
    done = False
    while not done:
        start = time.perf_counter()
        actions = policy(env, obs)
        decision_time += time.perf_counter() - start
        obs, _, dones, info = env.step(actions)
        done = bool(dones["__all__"])
        steps += 1
        for key in ("revenue", "profit", "total_cost", "inventory_violation_count", "unit_switch_count"):
            totals[key] += float(info.get(key, 0.0))
        satisfaction = float(info.get("demand_satisfaction_rate", 0.0))
        totals["demand_satisfaction_rate"] += max(0.0, min(1.0, satisfaction))
    if steps:
        totals["demand_satisfaction_rate"] /= steps
    totals["average_decision_time_ms"] = 1000.0 * decision_time / max(1, steps)
    return totals


def evaluate_methods(
    env_config: Mapping[str, Any],
    policies: Mapping[str, PolicyFn],
    eval_seeds: Iterable[int],
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    rows: list[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, float]] = {}
    seed_list = list(eval_seeds)
    for method, policy in policies.items():
        metrics_rows = []
        for seed in seed_list:
            metrics = evaluate_policy(env_config, policy, seed)
            metrics_rows.append(metrics)
            rows.append({"method": method, "seed": seed, **metrics})
        summary[method] = summarize_metrics(metrics_rows)
    return rows, summary


def summarize_metrics(metrics_rows: list[Mapping[str, float]]) -> Dict[str, float]:
    profits = [float(row["profit"]) for row in metrics_rows]
    revenues = [float(row["revenue"]) for row in metrics_rows]
    costs = [float(row["total_cost"]) for row in metrics_rows]
    violations = [float(row["inventory_violation_count"]) for row in metrics_rows]
    switches = [float(row["unit_switch_count"]) for row in metrics_rows]
    satisfaction = [float(row["demand_satisfaction_rate"]) for row in metrics_rows]
    decision_times = [float(row["average_decision_time_ms"]) for row in metrics_rows]
    return {
        "mean_profit": float(mean(profits)),
        "std_profit": float(pstdev(profits)) if len(profits) > 1 else 0.0,
        "min_profit": float(min(profits)),
        "max_profit": float(max(profits)),
        "cvar10_profit": cvar(profits, alpha=0.10),
        "mean_revenue": float(mean(revenues)),
        "mean_total_cost": float(mean(costs)),
        "mean_inventory_violations": float(mean(violations)),
        "mean_unit_switches": float(mean(switches)),
        "mean_demand_satisfaction_rate": float(mean(satisfaction)),
        "average_decision_time_ms": float(mean(decision_times)),
    }


def cvar(values: Iterable[float], alpha: float = 0.10) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return 0.0
    tail_count = max(1, int(np.ceil(len(sorted_values) * float(alpha))))
    return float(mean(sorted_values[:tail_count]))


def write_csv(path: str, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
