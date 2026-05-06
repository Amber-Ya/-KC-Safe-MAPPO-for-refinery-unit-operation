"""Reward computation for KC-Safe-MAPPO refinery scheduling."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


class RewardCalculator:
    """Compute global, local, and fused rewards from transition costs.

    The reward is designed so that its cumulative value over a full episode is
    comparable in magnitude to the MILP objective (profit maximisation).

    Key design decisions:
    - Revenue uses grade-level product prices × allocated product sales.
    - Crude cost uses purchase volume × average crude price.
    - Energy cost uses per-unit utility cost coefficients (same as MILP).
    - Inventory cost is accumulated per step, matching the MILP holding-cost
      summation over the horizon.
    - Shortage and violation penalties are tunable soft-constraint terms.
    """

    def __init__(self, env_config: Dict[str, Any], reward_config: Dict[str, float]):
        self.config = env_config
        self.reward_config = reward_config
        self.agents = list(env_config["agents"])
        self.utility_costs = env_config.get("utility_costs", {})
        self.inventory_nodes = env_config["inventory_nodes"]
        self.product_pools = env_config["product_pools"]
        self.product_pool_prices: Dict[str, float] = env_config.get("prices", {}).get(
            "product_pools", {}
        )
        # Scale inventory cost so that sum over horizon ≈ MILP inventory cost.
        self.horizon = max(1, int(env_config.get("time", {}).get("num_periods", 24)))

    def compute(
        self, state: Mapping[str, Any], transition_info: Mapping[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        unit_loads = transition_info["unit_loads"]
        previous_unit_loads = state["unit_loads"]
        inventories = transition_info["inventories"]
        product_pools_state = transition_info["product_pools"]
        shortage = transition_info["shortage"]
        product_delivery = transition_info.get("product_delivery", {})
        product_sales = transition_info.get("product_sales", {})

        # --- Revenue ---
        if product_sales:
            product_grades = self.config.get("prices", {}).get("product_grades", {})
            revenue = sum(
                float(product_sales.get(product, 0.0))
                * float(product_grades.get(product, {}).get("price", 0.0))
                for product in product_grades
            )
        else:
            revenue = sum(
                float(product_delivery.get(pool, 0.0)) * float(self.product_pool_prices.get(pool, 0.0))
                for pool in self.product_pools
            )

        # --- Costs ---
        crude_cost = float(transition_info.get("crude_purchase", 0.0)) * float(
            self.config.get("crude_average_price", 0.0)
        )
        energy_cost = sum(
            float(unit_loads[u]) * float(self.utility_costs.get(u, 0.0)) for u in self.agents
        )
        switch_cost = self.reward_config["switch_cost"] * sum(
            float(abs(float(unit_loads[u]) - float(previous_unit_loads.get(u, 0.0))) > 1e-6)
            for u in self.agents
        )
        shortage_cost = self.reward_config["shortage_cost"] * sum(
            float(v) for v in shortage.values()
        )

        # Inventory holding cost – full per-step cost (matching MILP where
        # inventory_cost × inventory_level is summed for every period).
        inventory_penalty = 0.0
        for node, amount in inventories.items():
            inventory_penalty += (
                float(self.inventory_nodes[node].get("inventory_cost", 0.0))
                * float(amount)
            )
        for pool, amount in product_pools_state.items():
            inventory_penalty += (
                float(self.product_pools[pool].get("inventory_cost", 0.0))
                * float(amount)
            )

        violation_penalty = self.reward_config["violation_cost"] * float(
            transition_info.get("violation_count", 0)
        )

        total_cost = (
            crude_cost + energy_cost + switch_cost + shortage_cost
            + inventory_penalty + violation_penalty
        )
        profit = revenue - total_cost
        reward_scale = max(1.0, float(self.reward_config.get("reward_scale", 1.0)))
        efficiency_bonus = self._crude_efficiency_bonus(revenue, crude_cost)
        global_reward = (profit + efficiency_bonus) / reward_scale

        # --- Per-agent local rewards ---
        local_rewards: Dict[str, float] = {}
        for unit in self.agents:
            local_energy = float(unit_loads[unit]) * float(self.utility_costs.get(unit, 0.0))
            local_switch = self.reward_config["switch_cost"] * float(
                abs(float(unit_loads[unit]) - float(previous_unit_loads.get(unit, 0.0))) > 1e-6
            )
            blocked = float(transition_info.get("blocked_load_by_unit", {}).get(unit, 0.0))
            local_rewards[unit] = -(
                local_energy + local_switch + self.reward_config["violation_cost"] * blocked
            ) / reward_scale

        alpha = float(self.reward_config.get("alpha", 0.7))
        rewards = {
            unit: alpha * global_reward + (1.0 - alpha) * local_rewards[unit]
            for unit in self.agents
        }
        metrics = {
            "global_reward": global_reward,
            "revenue": revenue,
            "profit": profit,
            "reward_profit": profit + efficiency_bonus,
            "crude_efficiency_bonus": efficiency_bonus,
            "total_cost": total_cost,
            "crude_cost": crude_cost,
            "energy_cost": energy_cost,
            "switch_cost": switch_cost,
            "shortage_cost": shortage_cost,
            "inventory_penalty": inventory_penalty,
            "violation_penalty": violation_penalty,
        }
        return rewards, metrics

    def _crude_efficiency_bonus(self, revenue: float, crude_cost: float) -> float:
        coef = float(self.reward_config.get("crude_efficiency_bonus", 0.0))
        if coef <= 0.0 or crude_cost <= 1e-8:
            return 0.0
        return coef * max(0.0, revenue / crude_cost - 1.0)
