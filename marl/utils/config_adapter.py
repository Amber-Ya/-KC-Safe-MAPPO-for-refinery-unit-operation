"""Adapter from the shared refinery case data to MARL environment data."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping


AGENTS = ["CDU1", "CDU2", "DFHC", "FCC1", "FCC2", "ROHU", "DHC"]


class ConfigAdapter:
    """Build a compact, MARL-friendly view of the existing config.py data."""

    def __init__(self, config_module: Any):
        self.config = config_module
        self.data: Mapping[str, Any] = config_module.CASE_DATA

    def get_agents(self) -> List[str]:
        return [agent for agent in AGENTS if agent in self.data["units"]]

    def get_units(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy({u: self.data["units"][u] for u in self.get_agents()})

    def get_inventory_nodes(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy(self.data.get("inventory_nodes", {}))

    def get_product_pools(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy(self.data.get("product_pools", {}))

    def get_routes(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "direct": deepcopy(self.data.get("direct_routes", [])),
            "buffer": deepcopy(self.data.get("buffer_routes", [])),
            "blending": deepcopy(self.data.get("blending_routes", [])),
        }

    def get_yields(self) -> Dict[str, Dict[str, Any]]:
        yields: Dict[str, Dict[str, Any]] = {}
        yields.update(deepcopy(self.data.get("cdu_yields_aggregated", {})))
        yields.update(deepcopy(self.data.get("secondary_unit_yields_aggregated", {})))
        return yields

    def get_demands(self) -> Dict[str, Dict[str, Any]]:
        time_horizon = max(1, int(self.data["time"]["num_periods"]))
        groups = deepcopy(self.data.get("demand_groups", {}))
        for group in groups.values():
            group["per_period_min"] = float(group.get("base_min_total", 0.0)) / time_horizon
        return groups

    def get_prices(self) -> Dict[str, Any]:
        product_grades = self.data.get("product_grades", {})
        product_pool_prices: Dict[str, float] = {}
        for pool in self.data.get("product_pools", {}):
            prices = [
                float(info["price"])
                for info in product_grades.values()
                if info.get("pool") == pool and "price" in info
            ]
            product_pool_prices[pool] = sum(prices) / len(prices) if prices else 0.0
        return {
            "crude": deepcopy(self.data.get("crude_price_base", {})),
            "product_grades": deepcopy(product_grades),
            "product_pools": product_pool_prices,
        }

    def get_utility_costs(self) -> Dict[str, float]:
        utility_names = list(self.data.get("utility_price", {}).keys())
        utility_price = self.data.get("utility_price", {})
        costs: Dict[str, float] = {}
        for unit in self.get_agents():
            coeffs = self.data.get("unit_utility_coefficients", {}).get(unit, [])
            cost = 0.0
            for idx, coeff in enumerate(coeffs):
                if idx < len(utility_names):
                    cost += float(coeff) * float(utility_price[utility_names[idx]])
            costs[unit] = cost
        return costs

    def build_env_config(self) -> Dict[str, Any]:
        crude_prices = self.data.get("crude_price_base", {})
        crude_avg_price = (
            sum(float(v) for v in crude_prices.values()) / len(crude_prices)
            if crude_prices
            else 0.0
        )
        crude_supply_max = sum(
            float(info.get("max", 0.0)) for info in self.data.get("crude_supply", {}).values()
        )
        component_pools = [
            name
            for name, info in self.data.get("inventory_nodes", {}).items()
            if info.get("type") == "component_pool"
        ]
        return {
            "case_name": self.data.get("case_name", "refinery_marl_case"),
            "agents": self.get_agents(),
            "time": deepcopy(self.data["time"]),
            "units": self.get_units(),
            "inventory_nodes": self.get_inventory_nodes(),
            "component_pools": component_pools,
            "product_pools": self.get_product_pools(),
            "routes": self.get_routes(),
            "yields": self.get_yields(),
            "demands": self.get_demands(),
            "prices": self.get_prices(),
            "utility_costs": self.get_utility_costs(),
            "crudes": deepcopy(self.data.get("crudes", [])),
            "crude_supply": deepcopy(self.data.get("crude_supply", {})),
            "crude_purchase_max_per_period": crude_supply_max,
            "crude_average_price": crude_avg_price,
            "unit_order": ["CDU1", "CDU2", "DFHC", "FCC1", "FCC2", "ROHU", "DHC"],
        }
