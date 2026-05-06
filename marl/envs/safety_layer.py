"""Knowledge-constrained safety layer for refinery unit actions."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from marl.configs.algo_config import LOAD_DELTA_ACTIONS, LOAD_LEVELS


class SafetyLayer:
    """Mask and repair discrete load-adjustment actions."""

    def __init__(self, env_config: Dict[str, Any]):
        self.config = env_config
        self.agents = list(env_config["agents"])
        self.units = env_config["units"]
        self.inventory_nodes = env_config["inventory_nodes"]
        self.product_pools = env_config["product_pools"]
        self.yields = env_config["yields"]
        self.load_levels = list(LOAD_LEVELS)
        self.load_delta_actions = dict(LOAD_DELTA_ACTIONS)

    def get_action_mask(self, state: Mapping[str, Any], agent_id: str) -> np.ndarray:
        """Return a binary mask over absolute load-level actions."""
        n_actions = len(self.load_levels)
        mask = np.ones(n_actions, dtype=np.float32)
        # Mask out levels that violate upstream / downstream feasibility.
        upstream = self._upstream_available(state, agent_id)
        downstream = self._downstream_available_capacity(state, agent_id)
        for idx in range(1, n_actions):  # level 0 (off) always allowed
            load = self.index_to_load(agent_id, idx)
            if upstream <= 1e-8 or downstream <= 1e-8:
                mask[idx] = 0.0
            elif load > upstream + 1e-6:
                mask[idx] = 0.0
        if mask.sum() <= 0.0:
            mask[0] = 1.0  # fallback: shut down
        return mask

    def repair_joint_action(
        self, state: Mapping[str, Any], raw_actions: Mapping[str, int]
    ) -> Dict[str, int]:
        repaired: Dict[str, int] = {}
        for agent in self.agents:
            action = int(raw_actions.get(agent, 0))
            mask = self.get_action_mask(state, agent)
            n = len(mask)
            action = max(0, min(action, n - 1))
            if mask[action] > 0.0:
                repaired[agent] = action
            else:
                # Fallback: find highest feasible load level.
                repaired[agent] = max(
                    (i for i in range(n) if mask[i] > 0.0), default=0
                )
        return repaired

    def action_to_load(self, unit: str, current_index: int, action: int) -> tuple[int, float]:
        """Absolute action: action IS the target load level index."""
        next_index = int(np.clip(action, 0, len(self.load_levels) - 1))
        return next_index, self.index_to_load(unit, next_index)

    def index_to_load(self, unit: str, index: int) -> float:
        frac = self.load_levels[int(index)]
        if frac <= 0.0:
            return 0.0
        info = self.units[unit]
        cap_min = float(info.get("capacity_min", 0.0))
        cap_max = float(info.get("capacity_max", 0.0))
        return cap_min + frac * (cap_max - cap_min)

    def repair_load(
        self,
        unit: str,
        proposed_load: float,
        state: Mapping[str, Any],
        proposed_loads: Mapping[str, float] | None = None,
    ) -> float:
        load = max(0.0, float(proposed_load))
        cap_min = float(self.units[unit].get("capacity_min", 0.0))
        cap_max = float(self.units[unit].get("capacity_max", 0.0))
        load = min(load, cap_max)
        if 0.0 < load < cap_min:
            load = cap_min

        upstream = self._upstream_available(state, unit, proposed_loads)
        downstream = self._downstream_available_capacity(state, unit)
        if upstream <= 1e-8 or downstream <= 1e-8:
            return 0.0
        return max(0.0, min(load, upstream, self._load_limit_from_downstream(unit, downstream)))

    def repair_loads(
        self, state: Mapping[str, Any], safe_actions: Mapping[str, int]
    ) -> tuple[Dict[str, int], Dict[str, float]]:
        proposed_indices: Dict[str, int] = {}
        proposed_loads: Dict[str, float] = {}
        for unit in self.agents:
            idx, load = self.action_to_load(
                unit, int(state["load_level_indices"][unit]), int(safe_actions[unit])
            )
            proposed_indices[unit] = idx
            proposed_loads[unit] = load

        repaired_loads: Dict[str, float] = {}
        for unit in self.agents:
            repaired_loads[unit] = self.repair_load(unit, proposed_loads[unit], state, proposed_loads)
        return proposed_indices, repaired_loads

    def _upstream_available(
        self,
        state: Mapping[str, Any],
        unit: str,
        proposed_loads: Mapping[str, float] | None = None,
    ) -> float:
        inv = state["inventories"]
        if unit in ("CDU1", "CDU2"):
            return float(inv.get("crude_tanks", 0.0)) + float(
                self.config.get("crude_purchase_max_per_period", 0.0)
            )
        if unit == "DFHC":
            direct = self._cdu_output("distillate_buffer", proposed_loads)
            return float(inv.get("distillate_buffer", 0.0)) + direct
        if unit in ("FCC1", "FCC2"):
            direct = self._cdu_output("fcc_feed_buffer", proposed_loads)
            return float(inv.get("fcc_feed_buffer", 0.0)) + 0.5 * direct
        if unit == "ROHU":
            return float(inv.get("residual_buffer", 0.0))
        if unit == "DHC":
            return float(inv.get("hydro_feed_buffer", 0.0)) + float(inv.get("distillate_buffer", 0.0))
        return 0.0

    def _downstream_available_capacity(self, state: Mapping[str, Any], unit: str) -> float:
        inv = state["inventories"]
        candidates = self._primary_output_nodes(unit) if unit.startswith("CDU") else self._secondary_output_nodes(unit)
        if not candidates:
            return 0.0
        available = 0.0
        for node in candidates:
            if node in self.inventory_nodes:
                available += float(self.inventory_nodes[node].get("max", 0.0)) - float(inv.get(node, 0.0))
        return max(0.0, available)

    def _load_limit_from_downstream(self, unit: str, downstream_capacity: float) -> float:
        tracked_yield = 0.0
        yields = self.yields.get(unit, {})
        if unit == "ROHU" and isinstance(yields, dict) and "residue_hydrotreating" in yields:
            yields = yields["residue_hydrotreating"]
        if isinstance(yields, dict):
            tracked_yield = sum(float(v) for k, v in yields.items() if k != "byproduct_or_untracked")
        if tracked_yield <= 1e-8:
            return float(self.units[unit].get("capacity_max", 0.0))
        return downstream_capacity / tracked_yield

    def _cdu_output(
        self, output_node: str, proposed_loads: Mapping[str, float] | None = None
    ) -> float:
        if proposed_loads is None:
            return 0.0
        amount = 0.0
        for cdu in ("CDU1", "CDU2"):
            amount += float(self.yields.get(cdu, {}).get(output_node, 0.0)) * float(proposed_loads.get(cdu, 0.0))
        return amount

    def _primary_output_nodes(self, unit: str) -> list[str]:
        if unit not in ("CDU1", "CDU2"):
            return []
        return ["naphtha_buffer", "distillate_buffer", "fcc_feed_buffer", "residual_buffer"]

    def _secondary_output_nodes(self, unit: str) -> list[str]:
        yields = self.yields.get(unit, {})
        if unit == "ROHU" and isinstance(yields, dict) and "residue_hydrotreating" in yields:
            yields = yields["residue_hydrotreating"]
        if not isinstance(yields, dict):
            return []
        return [
            node
            for node, coeff in yields.items()
            if node != "byproduct_or_untracked" and float(coeff) > 0.0
        ]
