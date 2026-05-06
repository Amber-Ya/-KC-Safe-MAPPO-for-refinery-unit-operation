"""Simplified component-pool to product-pool blending module."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


class BlendingModule:
    """Move aggregate component pools into final product pools and deliver demand."""

    COMPONENT_TO_PRODUCT = {
        "gasoline_component_pool": "gasoline_pool",
        "diesel_jet_component_pool": "diesel_pool",
        "lpg_component_pool": "lpg_pool",
    }

    def __init__(self, env_config: Dict[str, Any]):
        self.config = env_config
        self.inventory_nodes = env_config["inventory_nodes"]
        self.product_pools = env_config["product_pools"]

    def blend(self, state: Mapping[str, Any], demand: Mapping[str, float]) -> Dict[str, Any]:
        inventories = {k: float(v) for k, v in state["inventories"].items()}
        product_pools = {k: float(v) for k, v in state["product_pools"].items()}
        flows: Dict[Tuple[str, str], float] = {}
        delivery: Dict[str, float] = {}
        shortage: Dict[str, float] = {}

        for component_pool, product_pool in self.COMPONENT_TO_PRODUCT.items():
            pool_capacity = self._product_available_capacity(product_pools, product_pool)
            component_available = inventories.get(component_pool, 0.0)
            # Blend up to the available capacity (not capped by single-period
            # demand), matching MILP where blend amounts are free variables.
            blend_amount = min(component_available, pool_capacity)
            inventories[component_pool] = component_available - blend_amount
            product_pools[product_pool] = product_pools.get(product_pool, 0.0) + blend_amount
            flows[(component_pool, product_pool)] = blend_amount

        for product_pool, target in demand.items():
            min_inventory = float(self.product_pools[product_pool].get("min", 0.0))
            pool_level = product_pools.get(product_pool, 0.0)
            # Sell ALL available inventory above the pool minimum, matching
            # MILP behaviour where sell variables are bounded by inventory,
            # not by minimum demand.
            available_for_delivery = max(0.0, pool_level - min_inventory)
            product_pools[product_pool] = pool_level - available_for_delivery
            delivery[product_pool] = available_for_delivery
            # Shortage is measured against the per-period demand target.
            shortage[product_pool] = max(0.0, float(target) - available_for_delivery)

        self._clip_bounds(inventories, product_pools)
        total_demand = sum(float(v) for v in demand.values())
        total_delivery = sum(delivery.values())
        return {
            "inventories": inventories,
            "product_pools": product_pools,
            "blend_flows": flows,
            "product_delivery": delivery,
            "shortage": shortage,
            "demand_satisfaction_rate": total_delivery / total_demand if total_demand > 1e-8 else 1.0,
        }

    def check_quality_constraints(self, product: str, recipe: Mapping[str, float]) -> bool:
        return True

    def _product_available_capacity(self, product_pools: Mapping[str, float], pool: str) -> float:
        return max(0.0, float(self.product_pools[pool].get("max", 0.0)) - float(product_pools.get(pool, 0.0)))

    def _clip_bounds(self, inventories: Dict[str, float], product_pools: Dict[str, float]) -> None:
        for node, info in self.inventory_nodes.items():
            inventories[node] = min(float(info.get("max", inventories[node])), max(float(info.get("min", 0.0)), inventories[node]))
        for pool, info in self.product_pools.items():
            product_pools[pool] = min(float(info.get("max", product_pools[pool])), max(float(info.get("min", 0.0)), product_pools[pool]))
