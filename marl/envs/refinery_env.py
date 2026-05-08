"""Multi-agent refinery scheduling environment for KC-Safe-MAPPO."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from marl.configs.algo_config import LOAD_LEVELS, REWARD_CONFIG
from marl.envs.blending import BlendingModule
from marl.envs.reward import RewardCalculator
from marl.envs.routing import FlowRouter
from marl.envs.safety_layer import SafetyLayer


class RefinerySchedulingEnv:
    """Refinery multi-unit scheduling MDP with decentralized unit actions."""

    def __init__(
        self,
        env_config: Dict[str, Any],
        seed: int | None = None,
        reward_config: Dict[str, float] | None = None,
    ):
        self.config = env_config
        self.uncertainty_config = dict(env_config.get("uncertainty", {"enabled": False}))
        self.agents = list(env_config["agents"])
        self.num_agents = len(self.agents)
        self.horizon = int(env_config["time"]["num_periods"])
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.safety_layer = SafetyLayer(env_config)
        self.router = FlowRouter(env_config)
        self.blender = BlendingModule(env_config)
        self.reward_calculator = RewardCalculator(env_config, reward_config or REWARD_CONFIG)
        self.action_dim = len(LOAD_LEVELS)
        self.state: Dict[str, Any] = {}
        self.obs_dim = 0
        self.global_state_dim = 0

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed
        inventories = {
            node: float(info.get("init", 0.0))
            for node, info in self.config["inventory_nodes"].items()
        }
        product_pools = {
            pool: float(info.get("init", 0.0))
            for pool, info in self.config["product_pools"].items()
        }
        uncertainty = self._sample_uncertainty_trajectory()
        self.state = {
            "t": 0,
            "inventories": inventories,
            "product_pools": product_pools,
            "unit_loads": {agent: 0.0 for agent in self.agents},
            "last_unit_loads": {agent: 0.0 for agent in self.agents},
            "unit_availability": uncertainty["unit_availability"][0],
            "load_level_indices": {agent: 0 for agent in self.agents},
            "last_actions": {agent: 1 for agent in self.agents},
            "switch_flags": {agent: 0.0 for agent in self.agents},
            "uncertainty": uncertainty,
            "demand_multipliers": uncertainty["demand_multipliers"][0],
            "price_multipliers": uncertainty["price_multipliers"][0],
            "yield_multipliers": uncertainty["yield_multipliers"][0],
            "crude_price_multiplier": uncertainty["crude_price_multipliers"][0],
            "cumulative_delivery": {pool: 0.0 for pool in self.config["product_pools"]},
            "cumulative_product_sales": {
                product: 0.0
                for product in self.config.get("prices", {}).get("product_grades", {})
            },
        }
        obs = {agent: self.build_observation(agent) for agent in self.agents}
        self.obs_dim = len(next(iter(obs.values())))
        self.global_state_dim = len(self.get_global_state())
        return obs

    def step(
        self, actions: Mapping[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        raw_actions = {agent: int(actions.get(agent, 1)) for agent in self.agents}
        safe_actions = self.safety_layer.repair_joint_action(self.state, raw_actions)
        load_indices, safe_loads = self.safety_layer.repair_loads(self.state, safe_actions)
        previous_state = deepcopy(self.state)

        routing_info = self.router.route(self.state, safe_loads)
        demand = self._current_demand()
        demand_max = self._current_demand_max()
        blend_state = {
            "inventories": routing_info["inventories"],
            "product_pools": self.state["product_pools"],
            "cumulative_product_sales": self.state["cumulative_product_sales"],
        }
        blending_info = self.blender.blend(blend_state, demand, demand_max)

        actual_loads = routing_info["actual_loads"]
        switch_flags = {
            agent: float(abs(actual_loads[agent] - self.state["unit_loads"].get(agent, 0.0)) > 1e-8)
            for agent in self.agents
        }
        blocked_by_unit = {
            agent: max(0.0, float(safe_loads.get(agent, 0.0)) - float(actual_loads.get(agent, 0.0)))
            for agent in self.agents
        }
        transition_info = {
            "unit_loads": actual_loads,
            "inventories": blending_info["inventories"],
            "product_pools": blending_info["product_pools"],
            "shortage": blending_info["shortage"],
            "product_delivery": blending_info["product_delivery"],
            "product_sales": blending_info["product_sales"],
            "crude_purchase": routing_info["crude_purchase"],
            "violation_count": routing_info["violation_count"],
            "blocked_load_by_unit": blocked_by_unit,
        }
        rewards, reward_metrics = self.reward_calculator.compute(previous_state, transition_info)

        self.state["t"] += 1
        self.state["last_unit_loads"] = previous_state["unit_loads"]
        self.state["unit_loads"] = actual_loads
        self.state["inventories"] = blending_info["inventories"]
        self.state["product_pools"] = blending_info["product_pools"]
        self.state["load_level_indices"] = load_indices
        self.state["last_actions"] = safe_actions
        self.state["switch_flags"] = switch_flags
        for pool, amount in blending_info["product_delivery"].items():
            self.state["cumulative_delivery"][pool] += float(amount)
        for product, amount in blending_info["product_sales"].items():
            self.state["cumulative_product_sales"][product] += float(amount)

        done = self.state["t"] >= self.horizon
        if not done:
            self._advance_uncertainty_state()
        next_obs = {agent: self.build_observation(agent) for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        infos = {
            "raw_actions": raw_actions,
            "safe_actions": safe_actions,
            "unit_loads": actual_loads,
            "flows": routing_info["flows"],
            "blend_flows": blending_info["blend_flows"],
            "inventories": blending_info["inventories"],
            "product_pools": blending_info["product_pools"],
            "product_delivery": blending_info["product_delivery"],
            "product_sales": blending_info["product_sales"],
            "shortage": blending_info["shortage"],
            "demand": demand,
            "uncertainty": {
                "demand_multipliers": previous_state.get("demand_multipliers", {}),
                "price_multipliers": previous_state.get("price_multipliers", {}),
                "yield_multipliers": previous_state.get("yield_multipliers", {}),
                "crude_price_multiplier": previous_state.get("crude_price_multiplier", 1.0),
                "unit_availability": previous_state.get("unit_availability", {}),
            },
            "demand_satisfaction_rate": blending_info["demand_satisfaction_rate"],
            "inventory_violation_count": routing_info["violation_count"],
            "unit_switch_count": int(sum(switch_flags.values())),
            **reward_metrics,
        }
        return next_obs, rewards, dones, infos

    def get_global_state(self) -> np.ndarray:
        t_ratio = float(self.state.get("t", 0)) / max(1, self.horizon)
        values = [t_ratio]
        values.extend(self.state["unit_loads"][agent] / self._capacity_max(agent) for agent in self.agents)
        values.extend(self.state["last_unit_loads"][agent] / self._capacity_max(agent) for agent in self.agents)
        values.extend(self.state["unit_availability"][agent] for agent in self.agents)
        values.extend(self._normalized_inventory(node) for node in self.config["inventory_nodes"])
        values.extend(self._normalized_product_pool(pool) for pool in self.config["product_pools"])
        current_demand = self._current_demand()
        values.extend(current_demand[pool] / max(1.0, float(self.config["product_pools"][pool].get("max", 1.0))) for pool in self.config["product_pools"])
        values.extend(float(self.state.get("demand_multipliers", {}).get(pool, 1.0)) for pool in self.config["product_pools"])
        values.extend(self._pool_price_multiplier(pool) for pool in self.config["product_pools"])
        values.append(float(self.state.get("crude_price_multiplier", 1.0)))
        values.extend(self._unit_yield_multiplier(agent) for agent in self.agents)
        values.extend(float(self.state["last_actions"][agent]) / 2.0 for agent in self.agents)
        values.extend(float(self.state["switch_flags"][agent]) for agent in self.agents)
        return np.asarray(values, dtype=np.float32)

    def build_observation(self, agent_id: str) -> np.ndarray:
        upstream_nodes, downstream_nodes, neighbor_units, demand_pools = self._local_context(agent_id)
        upstream_set = set(upstream_nodes)
        downstream_set = set(downstream_nodes)
        demand_set = set(demand_pools)
        neighbor_set = set(neighbor_units)
        values = [
            self.state["unit_loads"][agent_id] / self._capacity_max(agent_id),
            self.state["last_unit_loads"][agent_id] / self._capacity_max(agent_id),
            self.state["unit_availability"][agent_id],
        ]
        for node in self.config["inventory_nodes"]:
            relevance = 1.0 if node in upstream_set or node in downstream_set else 0.0
            values.extend([self._normalized_inventory(node), relevance])
        for pool in self.config["product_pools"]:
            relevance = 1.0 if pool in demand_set else 0.0
            values.extend([self._normalized_product_pool(pool), relevance])
        demand = self._current_demand()
        for pool in self.config["product_pools"]:
            values.append(demand[pool] / max(1.0, float(self.config["product_pools"][pool].get("max", 1.0))))
            values.append(float(self.state.get("demand_multipliers", {}).get(pool, 1.0)))
            values.append(self._pool_price_multiplier(pool) if pool in demand_set else 1.0)
        values.append(float(self.state.get("crude_price_multiplier", 1.0)))
        values.append(self._unit_yield_multiplier(agent_id))
        for unit in self.agents:
            relevance = 1.0 if unit in neighbor_set or unit == agent_id else 0.0
            values.extend([self.state["unit_loads"][unit] / self._capacity_max(unit), relevance])
        values.append(float(self.state.get("t", 0)) / max(1, self.horizon))
        return np.asarray(values, dtype=np.float32)

    def get_action_mask(self, agent_id: str) -> np.ndarray:
        return self.safety_layer.get_action_mask(self.state, agent_id)

    def _current_demand(self) -> Dict[str, float]:
        demand = {pool: 0.0 for pool in self.config["product_pools"]}
        remaining_periods = max(1, self.horizon - int(self.state.get("t", 0)))
        for group in self.config.get("demands", {}).values():
            pool = group["pool"]
            multiplier = float(self.state.get("demand_multipliers", {}).get(pool, 1.0))
            total_min = float(group.get("base_min_total", 0.0)) * multiplier
            already_delivered = float(self.state.get("cumulative_delivery", {}).get(pool, 0.0))
            demand[pool] += max(0.0, total_min - already_delivered) / remaining_periods
        return demand

    def _current_demand_max(self) -> Dict[str, float]:
        """Remaining delivery budget per pool (mirrors MILP horizon-total demand_max)."""
        demand_max = {pool: float("inf") for pool in self.config["product_pools"]}
        for group in self.config.get("demands", {}).values():
            pool = group["pool"]
            total_max = float(group.get("demand_max_total", float("inf")))
            if total_max < float("inf"):
                total_max *= float(self.state.get("demand_multipliers", {}).get(pool, 1.0))
            already_delivered = self.state.get("cumulative_delivery", {}).get(pool, 0.0)
            demand_max[pool] = max(0.0, total_max - already_delivered)
        return demand_max

    def _sample_uncertainty_trajectory(self) -> Dict[str, Any]:
        products = self.config.get("prices", {}).get("product_grades", {})
        pools = list(self.config["product_pools"])
        enabled = bool(self.uncertainty_config.get("enabled", False))
        if not enabled:
            return {
                "demand_multipliers": [{pool: 1.0 for pool in pools} for _ in range(self.horizon)],
                "price_multipliers": [{product: 1.0 for product in products} for _ in range(self.horizon)],
                "yield_multipliers": [self._unit_yield_multiplier_defaults() for _ in range(self.horizon)],
                "crude_price_multipliers": [1.0 for _ in range(self.horizon)],
                "unit_availability": [{agent: 1.0 for agent in self.agents} for _ in range(self.horizon)],
            }

        demand_std = float(self.uncertainty_config.get("demand_std", 0.0))
        price_std = float(self.uncertainty_config.get("price_std", 0.0))
        yield_std = float(self.uncertainty_config.get("yield_std", 0.0))
        crude_std = float(self.uncertainty_config.get("crude_price_std", 0.0))
        outage_prob = float(self.uncertainty_config.get("unit_outage_prob", 0.0))
        derate_min = float(self.uncertainty_config.get("unit_derate_min", 0.6))

        demand_multipliers = []
        price_multipliers = []
        yield_multipliers = []
        crude_price_multipliers = []
        unit_availability = []
        for _ in range(self.horizon):
            demand_multipliers.append({
                pool: self._clipped_normal(1.0, demand_std, 0.65, 1.45)
                for pool in pools
            })
            price_multipliers.append({
                product: self._clipped_normal(1.0, price_std, 0.70, 1.35)
                for product in products
            })
            yield_multipliers.append(self._sample_yield_multipliers(yield_std))
            crude_price_multipliers.append(self._clipped_normal(1.0, crude_std, 0.75, 1.35))
            unit_availability.append({
                agent: (
                    float(self.rng.uniform(derate_min, 0.85))
                    if self.rng.random() < outage_prob
                    else 1.0
                )
                for agent in self.agents
            })

        return {
            "demand_multipliers": demand_multipliers,
            "price_multipliers": price_multipliers,
            "yield_multipliers": yield_multipliers,
            "crude_price_multipliers": crude_price_multipliers,
            "unit_availability": unit_availability,
        }

    def _advance_uncertainty_state(self) -> None:
        t = int(self.state["t"])
        uncertainty = self.state["uncertainty"]
        self.state["unit_availability"] = uncertainty["unit_availability"][t]
        self.state["demand_multipliers"] = uncertainty["demand_multipliers"][t]
        self.state["price_multipliers"] = uncertainty["price_multipliers"][t]
        self.state["yield_multipliers"] = uncertainty["yield_multipliers"][t]
        self.state["crude_price_multiplier"] = uncertainty["crude_price_multipliers"][t]

    def _clipped_normal(self, mean: float, std: float, low: float, high: float) -> float:
        if std <= 1e-12:
            return float(mean)
        return float(np.clip(self.rng.normal(mean, std), low, high))

    def _unit_yield_multiplier_defaults(self) -> Dict[str, Dict[str, float]]:
        multipliers: Dict[str, Dict[str, float]] = {}
        for unit, yields in self.config["yields"].items():
            if unit == "ROHU" and isinstance(yields, dict) and "residue_hydrotreating" in yields:
                flat_yields = yields["residue_hydrotreating"]
            else:
                flat_yields = yields
            if isinstance(flat_yields, dict):
                multipliers[unit] = {
                    node: 1.0
                    for node, coeff in flat_yields.items()
                    if node != "byproduct_or_untracked" and float(coeff) > 0.0
                }
        return multipliers

    def _sample_yield_multipliers(self, std: float) -> Dict[str, Dict[str, float]]:
        defaults = self._unit_yield_multiplier_defaults()
        if std <= 1e-12:
            return defaults
        return {
            unit: {
                node: self._clipped_normal(1.0, std, 0.75, 1.15)
                for node in nodes
            }
            for unit, nodes in defaults.items()
        }

    def _local_context(self, agent_id: str) -> tuple[list[str], list[str], list[str], list[str]]:
        if agent_id in ("CDU1", "CDU2"):
            return (
                ["crude_tanks"],
                ["naphtha_buffer", "distillate_buffer", "fcc_feed_buffer", "residual_buffer"],
                ["CDU1" if agent_id == "CDU2" else "CDU2"],
                ["gasoline_pool", "diesel_pool", "lpg_pool"],
            )
        if agent_id == "DFHC":
            return (
                ["distillate_buffer"],
                ["gasoline_component_pool", "diesel_jet_component_pool"],
                ["CDU1", "CDU2", "DHC"],
                ["diesel_pool"],
            )
        if agent_id in ("FCC1", "FCC2"):
            return (
                ["fcc_feed_buffer"],
                ["gasoline_component_pool", "diesel_jet_component_pool", "lpg_component_pool"],
                ["FCC2" if agent_id == "FCC1" else "FCC1"],
                ["gasoline_pool", "lpg_pool"],
            )
        if agent_id == "ROHU":
            return (
                ["residual_buffer"],
                ["hydro_feed_buffer", "diesel_jet_component_pool"],
                ["DHC"],
                ["diesel_pool"],
            )
        return (
            ["hydro_feed_buffer", "distillate_buffer"],
            ["gasoline_component_pool", "diesel_jet_component_pool", "lpg_component_pool"],
            ["ROHU", "DFHC"],
            ["diesel_pool", "gasoline_pool"],
        )

    def _normalized_inventory(self, node: str) -> float:
        info = self.config["inventory_nodes"][node]
        max_inv = max(1.0, float(info.get("max", 1.0)))
        return float(self.state["inventories"].get(node, 0.0)) / max_inv

    def _normalized_product_pool(self, pool: str) -> float:
        info = self.config["product_pools"][pool]
        max_inv = max(1.0, float(info.get("max", 1.0)))
        return float(self.state["product_pools"].get(pool, 0.0)) / max_inv

    def _capacity_max(self, unit: str) -> float:
        return max(1.0, float(self.config["units"][unit].get("capacity_max", 1.0)))

    def _pool_price_multiplier(self, pool: str) -> float:
        product_grades = self.config.get("prices", {}).get("product_grades", {})
        multipliers = [
            float(self.state.get("price_multipliers", {}).get(product, 1.0))
            for product, info in product_grades.items()
            if info.get("pool") == pool
        ]
        return float(sum(multipliers) / len(multipliers)) if multipliers else 1.0

    def _unit_yield_multiplier(self, unit: str) -> float:
        multipliers = [
            float(value)
            for value in self.state.get("yield_multipliers", {}).get(unit, {}).values()
        ]
        return float(sum(multipliers) / len(multipliers)) if multipliers else 1.0
