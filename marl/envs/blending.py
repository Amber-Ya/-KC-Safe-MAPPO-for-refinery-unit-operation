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
        self.product_grades = env_config.get("prices", {}).get("product_grades", {})
        self.big_m = float(env_config.get("big_m", 100.0))

    def blend(
        self,
        state: Mapping[str, Any],
        demand: Mapping[str, float],
        demand_max: Mapping[str, float] | None = None,
    ) -> Dict[str, Any]:
        inventories = {k: float(v) for k, v in state["inventories"].items()}
        product_pools = {k: float(v) for k, v in state["product_pools"].items()}
        cumulative_product_sales = {
            k: float(v) for k, v in state.get("cumulative_product_sales", {}).items()
        }
        flows: Dict[Tuple[str, str], float] = {}
        delivery: Dict[str, float] = {}
        product_sales: Dict[str, float] = {product: 0.0 for product in self.product_grades}
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
            available_for_delivery = max(0.0, pool_level - min_inventory)
            # Cap delivery by remaining horizon demand_max where the aggregate
            # pool has no unbounded grade to absorb extra sales.
            if demand_max is not None:
                pool_max = float(demand_max.get(product_pool, float("inf")))
                available_for_delivery = min(available_for_delivery, pool_max)
            allocated_sales = self._allocate_product_sales(
                product_pool,
                available_for_delivery,
                cumulative_product_sales,
            )
            delivered = sum(allocated_sales.values())
            for product, amount in allocated_sales.items():
                product_sales[product] = amount
            product_pools[product_pool] = pool_level - delivered
            delivery[product_pool] = delivered
            # Shortage is measured against the per-period demand target.
            shortage[product_pool] = max(0.0, float(target) - delivered)

        self._clip_bounds(inventories, product_pools)
        total_demand = sum(float(v) for v in demand.values())
        total_delivery = sum(delivery.values())
        return {
            "inventories": inventories,
            "product_pools": product_pools,
            "blend_flows": flows,
            "product_delivery": delivery,
            "product_sales": product_sales,
            "shortage": shortage,
            "demand_satisfaction_rate": total_delivery / total_demand if total_demand > 1e-8 else 1.0,
        }

    def check_quality_constraints(self, product: str, recipe: Mapping[str, float]) -> bool:
        return True

    def _product_available_capacity(self, product_pools: Mapping[str, float], pool: str) -> float:
        return max(0.0, float(self.product_pools[pool].get("max", 0.0)) - float(product_pools.get(pool, 0.0)))

    def _allocate_product_sales(
        self,
        product_pool: str,
        available: float,
        cumulative_product_sales: Mapping[str, float],
    ) -> Dict[str, float]:
        remaining_available = max(0.0, float(available))
        products = [
            product for product, info in self.product_grades.items()
            if info.get("pool") == product_pool
        ]
        sales = {product: 0.0 for product in products}

        # First satisfy grade-level minimum demand.  The MILP enforces these as
        # horizon-total constraints, so cumulative progress is the right state.
        for product in sorted(products, key=lambda p: float(self.product_grades[p].get("price", 0.0))):
            if remaining_available <= 1e-8:
                break
            info = self.product_grades[product]
            remaining_min = max(
                0.0,
                float(info.get("demand_min", 0.0))
                - float(cumulative_product_sales.get(product, 0.0)),
            )
            amount = min(remaining_available, remaining_min, self._remaining_product_cap(product, cumulative_product_sales))
            sales[product] += amount
            remaining_available -= amount

        # Then sell surplus to the most profitable grades, respecting real
        # finite caps while treating BIG_M exactly as the Gurobi model does.
        for product in sorted(products, key=lambda p: float(self.product_grades[p].get("price", 0.0)), reverse=True):
            if remaining_available <= 1e-8:
                break
            cap = self._remaining_product_cap(product, cumulative_product_sales) - sales[product]
            amount = min(remaining_available, max(0.0, cap))
            sales[product] += amount
            remaining_available -= amount
        return sales

    def _remaining_product_cap(
        self,
        product: str,
        cumulative_product_sales: Mapping[str, float],
    ) -> float:
        demand_max = float(self.product_grades[product].get("demand_max", self.big_m))
        if demand_max >= 0.999 * self.big_m:
            return float("inf")
        return max(0.0, demand_max - float(cumulative_product_sales.get(product, 0.0)))

    def _clip_bounds(self, inventories: Dict[str, float], product_pools: Dict[str, float]) -> None:
        for node, info in self.inventory_nodes.items():
            inventories[node] = min(float(info.get("max", inventories[node])), max(float(info.get("min", 0.0)), inventories[node]))
        for pool, info in self.product_pools.items():
            product_pools[pool] = min(float(info.get("max", product_pools[pool])), max(float(info.get("min", 0.0)), product_pools[pool]))
