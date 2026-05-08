"""Rule-based refinery material routing with direct and buffer paths."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


FlowKey = Tuple[str, str]


class FlowRouter:
    """Route unit loads through configured refinery topology."""

    def __init__(self, env_config: Dict[str, Any]):
        self.config = env_config
        self.yields = env_config["yields"]
        self.inventory_nodes = env_config["inventory_nodes"]

    def route(self, state: Mapping[str, Any], safe_loads: Mapping[str, float]) -> Dict[str, Any]:
        inv = {k: float(v) for k, v in state["inventories"].items()}
        flows: Dict[FlowKey, float] = {}
        actual_loads = {u: 0.0 for u in safe_loads}
        violation_count = 0
        blocked_load = 0.0

        cdu_outputs = {
            "distillate_buffer": 0.0,
            "fcc_feed_buffer": 0.0,
        }
        # CDU crude supply: purchase crude to meet demand directly (matching
        # MILP where crude_feed variables go straight to CDU without a
        # bottleneck storage).  crude_tanks acts as overflow buffer only.
        crude_need = float(safe_loads.get("CDU1", 0.0)) + float(safe_loads.get("CDU2", 0.0))
        crude_available_in_tanks = inv.get("crude_tanks", 0.0)
        crude_deficit = max(0.0, crude_need - crude_available_in_tanks)
        crude_purchase = min(
            crude_deficit,
            float(self.config.get("crude_purchase_max_per_period", 0.0)),
        )
        # Total crude available for CDUs this period:
        total_crude_available = crude_available_in_tanks + crude_purchase
        # Distribute to CDUs (use tanks first, then purchased crude).
        inv["crude_tanks"] = 0.0  # will refill with any surplus below
        flows[("crude_supply", "crude_tanks")] = crude_purchase

        for cdu in ("CDU1", "CDU2"):
            load = min(float(safe_loads.get(cdu, 0.0)), total_crude_available)
            actual_loads[cdu] = load
            total_crude_available -= load
            flows[("crude_tanks", cdu)] = load
            for node in ["naphtha_buffer", "distillate_buffer", "fcc_feed_buffer", "residual_buffer"]:
                amount = self._yield(cdu, node, state) * load
                if node in ("distillate_buffer", "fcc_feed_buffer"):
                    cdu_outputs[node] += amount
                else:
                    accepted = self._add_inventory(inv, node, amount)
                    violation_count += int(accepted + 1e-8 < amount)
                    flows[(cdu, node)] = accepted
        # Any remaining crude goes back to tanks.
        inv["crude_tanks"] = min(
            total_crude_available,
            float(self.inventory_nodes.get("crude_tanks", {}).get("max", 60.0)),
        )

        dfhc_load, dfhc_buffer_use, dfhc_direct_use = self._consume_with_direct(
            inv, "distillate_buffer", cdu_outputs["distillate_buffer"], float(safe_loads.get("DFHC", 0.0))
        )
        actual_loads["DFHC"] = dfhc_load
        flows[("distillate_buffer", "DFHC")] = dfhc_buffer_use
        flows[("CDU_direct_distillate", "DFHC")] = dfhc_direct_use
        remaining_distillate = cdu_outputs["distillate_buffer"] - dfhc_direct_use
        flows[("CDU_distillate_overflow", "distillate_buffer")] = self._add_inventory(inv, "distillate_buffer", remaining_distillate)

        fcc_direct_remaining = cdu_outputs["fcc_feed_buffer"]
        for fcc in ("FCC1", "FCC2"):
            requested = float(safe_loads.get(fcc, 0.0))
            direct_share = min(fcc_direct_remaining, requested)
            fcc_direct_remaining -= direct_share
            residual_request = requested - direct_share
            buffer_use = min(inv.get("fcc_feed_buffer", 0.0), residual_request)
            inv["fcc_feed_buffer"] -= buffer_use
            actual_loads[fcc] = direct_share + buffer_use
            flows[("CDU_direct_vgo", fcc)] = direct_share
            flows[("fcc_feed_buffer", fcc)] = buffer_use
        flows[("CDU_vgo_overflow", "fcc_feed_buffer")] = self._add_inventory(inv, "fcc_feed_buffer", fcc_direct_remaining)

        rohu_load = min(float(safe_loads.get("ROHU", 0.0)), inv.get("residual_buffer", 0.0))
        inv["residual_buffer"] -= rohu_load
        actual_loads["ROHU"] = rohu_load
        flows[("residual_buffer", "ROHU")] = rohu_load

        dhc_requested = float(safe_loads.get("DHC", 0.0))
        hydro_use = min(inv.get("hydro_feed_buffer", 0.0), dhc_requested)
        inv["hydro_feed_buffer"] -= hydro_use
        distillate_to_dhc = min(inv.get("distillate_buffer", 0.0), dhc_requested - hydro_use)
        inv["distillate_buffer"] -= distillate_to_dhc
        actual_loads["DHC"] = hydro_use + distillate_to_dhc
        flows[("hydro_feed_buffer", "DHC")] = hydro_use
        flows[("distillate_buffer", "DHC")] = distillate_to_dhc

        for unit in ("DFHC", "FCC1", "FCC2", "ROHU", "DHC"):
            self._route_secondary_outputs(unit, actual_loads.get(unit, 0.0), inv, flows, state)

        naphtha_to_component = min(
            inv.get("naphtha_buffer", 0.0),
            self._available_capacity(inv, "gasoline_component_pool"),
        )
        inv["naphtha_buffer"] -= naphtha_to_component
        accepted = self._add_inventory(inv, "gasoline_component_pool", naphtha_to_component)
        flows[("naphtha_buffer", "gasoline_component_pool")] = accepted

        for unit, requested in safe_loads.items():
            blocked_load += max(0.0, float(requested) - actual_loads.get(unit, 0.0))
        violation_count += int(blocked_load > 1e-7)
        self._clip_inventory_bounds(inv)
        return {
            "flows": flows,
            "inventories": inv,
            "actual_loads": actual_loads,
            "crude_purchase": crude_purchase,
            "blocked_load": blocked_load,
            "violation_count": violation_count,
        }

    def _consume_with_direct(
        self, inv: Dict[str, float], buffer_node: str, direct_available: float, requested: float
    ) -> tuple[float, float, float]:
        direct_use = min(max(0.0, direct_available), requested)
        buffer_use = min(inv.get(buffer_node, 0.0), requested - direct_use)
        inv[buffer_node] = inv.get(buffer_node, 0.0) - buffer_use
        return direct_use + buffer_use, buffer_use, direct_use

    def _route_secondary_outputs(
        self,
        unit: str,
        load: float,
        inv: Dict[str, float],
        flows: Dict[FlowKey, float],
        state: Mapping[str, Any],
    ) -> None:
        yields = self.yields.get(unit, {})
        if unit == "ROHU" and isinstance(yields, dict) and "residue_hydrotreating" in yields:
            yields = yields["residue_hydrotreating"]
        if not isinstance(yields, dict):
            return
        for node, coeff in yields.items():
            if node == "byproduct_or_untracked":
                continue
            amount = self._yield(unit, node, state) * load
            accepted = self._add_inventory(inv, node, amount)
            flows[(unit, node)] = accepted

    def _yield(self, unit: str, node: str, state: Mapping[str, Any]) -> float:
        yields = self.yields.get(unit, {})
        if unit == "ROHU" and isinstance(yields, dict) and "residue_hydrotreating" in yields:
            yields = yields["residue_hydrotreating"]
        if not isinstance(yields, dict):
            return 0.0
        multiplier = (
            state.get("yield_multipliers", {})
            .get(unit, {})
            .get(node, 1.0)
        )
        return float(yields.get(node, 0.0)) * float(multiplier)

    def _add_inventory(self, inv: Dict[str, float], node: str, amount: float) -> float:
        amount = max(0.0, float(amount))
        accepted = min(amount, self._available_capacity(inv, node))
        inv[node] = inv.get(node, 0.0) + accepted
        return accepted

    def _available_capacity(self, inv: Mapping[str, float], node: str) -> float:
        max_inv = float(self.inventory_nodes[node].get("max", 0.0))
        return max(0.0, max_inv - float(inv.get(node, 0.0)))

    def _clip_inventory_bounds(self, inv: Dict[str, float]) -> None:
        for node, info in self.inventory_nodes.items():
            inv[node] = min(float(info.get("max", inv.get(node, 0.0))), max(float(info.get("min", 0.0)), inv.get(node, 0.0)))
