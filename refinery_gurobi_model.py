# -*- coding: utf-8 -*-
"""Gurobi model for the refinery multi-unit operation scheduling case.

The case data in config.py supports a full aggregate MILP: crude purchase and
storage, unit operation and modes, direct/buffer routing, component/product
pool inventory, product demand, and economic objective. A small optional
MIQCP extension can be enabled for crude blend qualities at CDU feeds.

Run:
    python3 refinery_gurobi_model.py --validate-only
    python3 refinery_gurobi_model.py --solve --result-dir results
    python3 refinery_gurobi_model.py --solve --quality-mode bilinear-crude
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from config import BIG_M, CASE_DATA


CaseData = Dict[str, Any]


@dataclass
class ModelBundle:
    model: Any
    variables: Dict[str, Any]
    sets: Dict[str, Any]
    warnings: List[str]


def validate_case_data(data: CaseData) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings) for the supplied case data."""
    errors: List[str] = []
    warnings: List[str] = []

    required = [
        "time",
        "crudes",
        "crude_supply",
        "crude_price_base",
        "units",
        "inventory_nodes",
        "product_pools",
        "cdu_yields_aggregated",
        "secondary_unit_yields_aggregated",
        "product_grades",
    ]
    for key in required:
        if key not in data:
            errors.append(f"Missing required CASE_DATA key: {key}")

    if errors:
        return errors, warnings

    for crude in data["crudes"]:
        if crude not in data["crude_supply"]:
            errors.append(f"Missing crude_supply for crude {crude}")
        if crude not in data["crude_price_base"]:
            errors.append(f"Missing crude_price_base for crude {crude}")

    units = data["units"]
    cdu_units = [u for u, info in units.items() if info["type"] == "primary_distillation"]
    for u in cdu_units:
        if u not in data["cdu_yields_aggregated"]:
            errors.append(f"Missing CDU yield table for {u}")

    for u, yields in data["cdu_yields_aggregated"].items():
        total = sum(v for k, v in yields.items() if k != "byproduct_or_untracked")
        byproduct = yields.get("byproduct_or_untracked", 0.0)
        if total > 1.0 + 1e-6 or total + byproduct > 1.0 + 1e-6:
            errors.append(f"CDU yields for {u} exceed 1.0")

    for u, yields in data["secondary_unit_yields_aggregated"].items():
        if u == "ROHU" and all(isinstance(v, Mapping) for v in yields.values()):
            for mode, mode_yields in yields.items():
                total = sum(v for k, v in mode_yields.items() if k != "byproduct_or_untracked")
                byproduct = mode_yields.get("byproduct_or_untracked", 0.0)
                if total > 1.0 + 1e-6 or total + byproduct > 1.0 + 1e-6:
                    errors.append(f"Secondary yields for {u}/{mode} exceed 1.0")
        else:
            total = sum(v for k, v in yields.items() if k != "byproduct_or_untracked")
            byproduct = yields.get("byproduct_or_untracked", 0.0)
            if total > 1.0 + 1e-6 or total + byproduct > 1.0 + 1e-6:
                errors.append(f"Secondary yields for {u} exceed 1.0")

    if data.get("product_quality_specs") and not data.get("component_quality"):
        warnings.append(
            "Product quality specs are present, but component_quality data is not. "
            "The aggregate MILP will not enforce final product quality constraints."
        )

    for u, info in units.items():
        if len(info.get("modes", [])) > 1 and u != "ROHU":
            warnings.append(
                f"{u} has multiple modes, but no mode-specific yield/cost table is supplied; "
                "mode binaries will enforce exclusivity only."
            )

    return errors, warnings


def _import_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "gurobipy is not installed in this Python environment. Install Gurobi "
            "and gurobipy, then run this script again."
        ) from exc
    return gp, GRB


def _periods(data: CaseData) -> range:
    return range(int(data["time"]["num_periods"]))


def _unit_sets(data: CaseData) -> Tuple[List[str], List[str], List[str]]:
    units = data["units"]
    cdus = [u for u, info in units.items() if info["type"] == "primary_distillation"]
    secondary = [u for u in units if u not in cdus]
    fcc_units = [u for u in secondary if units[u]["type"] == "fluid_catalytic_cracking"]
    return cdus, secondary, fcc_units


def _route_has(data: CaseData, source: str, target: str) -> bool:
    for route_group in ("direct_routes", "buffer_routes", "blending_routes"):
        for route in data.get(route_group, []):
            if route.get("from") == source and route.get("to") == target:
                return True
    return False


def build_refinery_model(
    data: Optional[CaseData] = None,
    *,
    quality_mode: str = "milp",
    model_name: Optional[str] = None,
) -> ModelBundle:
    """Build the refinery scheduling optimization model.

    quality_mode:
        "milp"            Build the aggregate linear model.
        "bilinear-crude"  Add CDU crude blend quality identities:
                           q_mix[u,q,t] * charge[u,t] = sum_c q[c] feed[c,u,t].
                           This turns the model into a nonconvex MIQCP.
    """
    data = data or CASE_DATA
    errors, warnings = validate_case_data(data)
    if errors:
        raise ValueError("Invalid case data:\n" + "\n".join(errors))

    gp, GRB = _import_gurobi()

    name = model_name or data.get("case_name", "refinery_multi_unit_schedule")
    model = gp.Model(name)

    T = list(_periods(data))
    crudes = list(data["crudes"])
    units = list(data["units"].keys())
    cdus, secondary_units, fcc_units = _unit_sets(data)
    inv_nodes = list(data["inventory_nodes"].keys())
    non_crude_inventory_nodes = [n for n in inv_nodes if n != "crude_tanks"]
    component_pools = [
        n
        for n, info in data["inventory_nodes"].items()
        if info.get("type") == "component_pool"
    ]
    product_pools = list(data["product_pools"].keys())
    products = list(data["product_grades"].keys())
    utility_names = list(data.get("utility_price", {}).keys())

    sets = {
        "T": T,
        "crudes": crudes,
        "units": units,
        "cdus": cdus,
        "secondary_units": secondary_units,
        "fcc_units": fcc_units,
        "inventory_nodes": inv_nodes,
        "non_crude_inventory_nodes": non_crude_inventory_nodes,
        "component_pools": component_pools,
        "product_pools": product_pools,
        "products": products,
        "utility_names": utility_names,
    }

    # Core operating variables.
    crude_buy = model.addVars(crudes, T, lb=0.0, name="crude_buy")
    crude_inv = model.addVars(crudes, T, lb=0.0, name="crude_inv")
    crude_feed = model.addVars(crudes, cdus, T, lb=0.0, name="crude_feed")
    charge = model.addVars(units, T, lb=0.0, name="unit_charge")
    unit_on = model.addVars(units, T, vtype=GRB.BINARY, name="unit_on")

    mode_on = {}
    mode_charge = {}
    for u in units:
        modes = data["units"][u].get("modes", ["normal"])
        mode_on[u] = model.addVars(modes, T, vtype=GRB.BINARY, name=f"mode_on[{u}]")
        mode_charge[u] = model.addVars(modes, T, lb=0.0, name=f"mode_charge[{u}]")

    # CDU route split variables.
    cdu_to_naphtha_buffer = model.addVars(cdus, T, lb=0.0, name="cdu_to_naphtha_buffer")
    cdu_to_distillate_buffer = model.addVars(cdus, T, lb=0.0, name="cdu_to_distillate_buffer")
    cdu_to_dfhc = model.addVars(cdus, T, lb=0.0, name="cdu_to_dfhc")
    cdu_to_fcc_feed_buffer = model.addVars(cdus, T, lb=0.0, name="cdu_to_fcc_feed_buffer")
    cdu_to_fcc = model.addVars(cdus, fcc_units, T, lb=0.0, name="cdu_to_fcc")
    cdu_to_residual_buffer = model.addVars(cdus, T, lb=0.0, name="cdu_to_residual_buffer")

    # Buffer withdrawal variables.
    naphtha_to_gasoline_component = model.addVars(T, lb=0.0, name="naphtha_to_gasoline_component")
    distillate_to_dfhc = model.addVars(T, lb=0.0, name="distillate_buffer_to_dfhc")
    distillate_to_dhc = model.addVars(T, lb=0.0, name="distillate_buffer_to_dhc")
    fcc_buffer_to_fcc = model.addVars(fcc_units, T, lb=0.0, name="fcc_buffer_to_fcc")
    residual_to_rohu = model.addVars(T, lb=0.0, name="residual_buffer_to_rohu")
    hydro_feed_to_dhc = model.addVars(T, lb=0.0, name="hydro_feed_buffer_to_dhc")

    # Pool variables.
    inventory = model.addVars(non_crude_inventory_nodes, T, lb=0.0, name="inventory")
    product_inventory = model.addVars(product_pools, T, lb=0.0, name="product_inventory")
    blend = model.addVars(component_pools, product_pools, T, lb=0.0, name="blend_to_product_pool")
    sell = model.addVars(products, T, lb=0.0, name="product_sale")

    # Crude supply bounds and inventory.
    crude_tank = data["inventory_nodes"]["crude_tanks"]
    for c in crudes:
        supply = data["crude_supply"][c]
        for t in T:
            crude_buy[c, t].LB = float(supply.get("min", 0.0))
            crude_buy[c, t].UB = float(supply.get("max", BIG_M))
            previous = 0.0 if t == T[0] else crude_inv[c, T[T.index(t) - 1]]
            model.addConstr(
                crude_inv[c, t]
                == previous + crude_buy[c, t] - gp.quicksum(crude_feed[c, u, t] for u in cdus),
                name=f"crude_balance[{c},{t}]",
            )

    for t in T:
        crude_total = gp.quicksum(crude_inv[c, t] for c in crudes)
        model.addConstr(crude_total <= crude_tank["max"], name=f"crude_tank_max[{t}]")
        model.addConstr(crude_total >= crude_tank["min"], name=f"crude_tank_min[{t}]")

    # Unit capacity, mode selection, and charge identities.
    for u in units:
        info = data["units"][u]
        cap_min = float(info.get("capacity_min", 0.0))
        cap_max = float(info.get("capacity_max", BIG_M))
        modes = info.get("modes", ["normal"])
        for t in T:
            model.addConstr(charge[u, t] <= cap_max * unit_on[u, t], name=f"unit_cap_max[{u},{t}]")
            model.addConstr(charge[u, t] >= cap_min * unit_on[u, t], name=f"unit_cap_min[{u},{t}]")
            model.addConstr(
                gp.quicksum(mode_on[u][m, t] for m in modes) == unit_on[u, t],
                name=f"one_mode_if_on[{u},{t}]",
            )
            model.addConstr(
                gp.quicksum(mode_charge[u][m, t] for m in modes) == charge[u, t],
                name=f"mode_charge_sum[{u},{t}]",
            )
            for m in modes:
                model.addConstr(
                    mode_charge[u][m, t] <= cap_max * mode_on[u][m, t],
                    name=f"mode_cap_max[{u},{m},{t}]",
                )
                model.addConstr(
                    mode_charge[u][m, t] >= cap_min * mode_on[u][m, t],
                    name=f"mode_cap_min[{u},{m},{t}]",
                )

    for u in cdus:
        for t in T:
            model.addConstr(
                charge[u, t] == gp.quicksum(crude_feed[c, u, t] for c in crudes),
                name=f"cdu_charge_feed[{u},{t}]",
            )

    # CDU product split balances.
    for u in cdus:
        y = data["cdu_yields_aggregated"][u]
        for t in T:
            model.addConstr(
                cdu_to_naphtha_buffer[u, t] == y["naphtha_buffer"] * charge[u, t],
                name=f"cdu_naphtha_yield[{u},{t}]",
            )
            model.addConstr(
                cdu_to_distillate_buffer[u, t] + cdu_to_dfhc[u, t]
                == y["distillate_buffer"] * charge[u, t],
                name=f"cdu_distillate_split[{u},{t}]",
            )
            model.addConstr(
                cdu_to_fcc_feed_buffer[u, t] + gp.quicksum(cdu_to_fcc[u, f, t] for f in fcc_units)
                == y["fcc_feed_buffer"] * charge[u, t],
                name=f"cdu_fcc_feed_split[{u},{t}]",
            )
            model.addConstr(
                cdu_to_residual_buffer[u, t] == y["residual_buffer"] * charge[u, t],
                name=f"cdu_residue_yield[{u},{t}]",
            )

    # Secondary unit input balances.
    for t in T:
        if "DFHC" in units:
            model.addConstr(
                charge["DFHC", t]
                == gp.quicksum(cdu_to_dfhc[u, t] for u in cdus) + distillate_to_dfhc[t],
                name=f"dfhc_feed_balance[{t}]",
            )
        for f in fcc_units:
            model.addConstr(
                charge[f, t] == gp.quicksum(cdu_to_fcc[u, f, t] for u in cdus) + fcc_buffer_to_fcc[f, t],
                name=f"fcc_feed_balance[{f},{t}]",
            )
        if "ROHU" in units:
            model.addConstr(charge["ROHU", t] == residual_to_rohu[t], name=f"rohu_feed_balance[{t}]")
        if "DHC" in units:
            model.addConstr(
                charge["DHC", t] == hydro_feed_to_dhc[t] + distillate_to_dhc[t],
                name=f"dhc_feed_balance[{t}]",
            )

    # Secondary output expressions keyed by (output_node, t).
    secondary_output: MutableMapping[Tuple[str, int], Any] = {}
    for node in [
        "gasoline_component_pool",
        "diesel_jet_component_pool",
        "lpg_component_pool",
        "hydro_feed_buffer",
    ]:
        for t in T:
            secondary_output[node, t] = gp.LinExpr()

    for u in secondary_units:
        yields = data["secondary_unit_yields_aggregated"].get(u, {})
        modes = data["units"][u].get("modes", ["normal"])
        for t in T:
            if u == "ROHU" and all(isinstance(v, Mapping) for v in yields.values()):
                for m in modes:
                    mode_yields = yields.get(m, {})
                    for node in secondary_output:
                        out_node, period = node
                        if period == t:
                            secondary_output[out_node, t] += mode_yields.get(out_node, 0.0) * mode_charge[u][m, t]
            else:
                for node_name in ["gasoline_component_pool", "diesel_jet_component_pool", "lpg_component_pool", "hydro_feed_buffer"]:
                    secondary_output[node_name, t] += yields.get(node_name, 0.0) * charge[u, t]

    # Inventory balances for intermediate buffers and component pools.
    for t in T:
        prev_t = None if t == T[0] else T[T.index(t) - 1]

        def prev_inventory(node_name: str):
            init = float(data["inventory_nodes"][node_name].get("init", 0.0))
            return init if prev_t is None else inventory[node_name, prev_t]

        model.addConstr(
            inventory["naphtha_buffer", t]
            == prev_inventory("naphtha_buffer")
            + gp.quicksum(cdu_to_naphtha_buffer[u, t] for u in cdus)
            - naphtha_to_gasoline_component[t],
            name=f"inv_balance[naphtha_buffer,{t}]",
        )
        model.addConstr(
            inventory["distillate_buffer", t]
            == prev_inventory("distillate_buffer")
            + gp.quicksum(cdu_to_distillate_buffer[u, t] for u in cdus)
            - distillate_to_dfhc[t]
            - distillate_to_dhc[t],
            name=f"inv_balance[distillate_buffer,{t}]",
        )
        model.addConstr(
            inventory["fcc_feed_buffer", t]
            == prev_inventory("fcc_feed_buffer")
            + gp.quicksum(cdu_to_fcc_feed_buffer[u, t] for u in cdus)
            - gp.quicksum(fcc_buffer_to_fcc[f, t] for f in fcc_units),
            name=f"inv_balance[fcc_feed_buffer,{t}]",
        )
        model.addConstr(
            inventory["residual_buffer", t]
            == prev_inventory("residual_buffer")
            + gp.quicksum(cdu_to_residual_buffer[u, t] for u in cdus)
            - residual_to_rohu[t],
            name=f"inv_balance[residual_buffer,{t}]",
        )
        model.addConstr(
            inventory["hydro_feed_buffer", t]
            == prev_inventory("hydro_feed_buffer")
            + secondary_output["hydro_feed_buffer", t]
            - hydro_feed_to_dhc[t],
            name=f"inv_balance[hydro_feed_buffer,{t}]",
        )
        model.addConstr(
            inventory["gasoline_component_pool", t]
            == prev_inventory("gasoline_component_pool")
            + naphtha_to_gasoline_component[t]
            + secondary_output["gasoline_component_pool", t]
            - gp.quicksum(blend["gasoline_component_pool", p, t] for p in product_pools),
            name=f"inv_balance[gasoline_component_pool,{t}]",
        )
        model.addConstr(
            inventory["diesel_jet_component_pool", t]
            == prev_inventory("diesel_jet_component_pool")
            + secondary_output["diesel_jet_component_pool", t]
            - gp.quicksum(blend["diesel_jet_component_pool", p, t] for p in product_pools),
            name=f"inv_balance[diesel_jet_component_pool,{t}]",
        )
        model.addConstr(
            inventory["lpg_component_pool", t]
            == prev_inventory("lpg_component_pool")
            + secondary_output["lpg_component_pool", t]
            - gp.quicksum(blend["lpg_component_pool", p, t] for p in product_pools),
            name=f"inv_balance[lpg_component_pool,{t}]",
        )

    # Inventory bounds from data.
    for node in non_crude_inventory_nodes:
        info = data["inventory_nodes"][node]
        for t in T:
            inventory[node, t].LB = float(info.get("min", 0.0))
            inventory[node, t].UB = float(info.get("max", BIG_M))

    # Allow only configured component-to-product-pool blending routes.
    for cpool in component_pools:
        for ppool in product_pools:
            if not _route_has(data, cpool, ppool):
                for t in T:
                    blend[cpool, ppool, t].UB = 0.0

    # Product pool balances and bounds.
    products_by_pool: Dict[str, List[str]] = {pool: [] for pool in product_pools}
    for product, info in data["product_grades"].items():
        products_by_pool[info["pool"]].append(product)

    for pool in product_pools:
        info = data["product_pools"][pool]
        for t in T:
            previous = float(info.get("init", 0.0)) if t == T[0] else product_inventory[pool, T[T.index(t) - 1]]
            model.addConstr(
                product_inventory[pool, t]
                == previous
                + gp.quicksum(blend[cpool, pool, t] for cpool in component_pools)
                - gp.quicksum(sell[p, t] for p in products_by_pool[pool]),
                name=f"product_pool_balance[{pool},{t}]",
            )
            product_inventory[pool, t].LB = float(info.get("min", 0.0))
            product_inventory[pool, t].UB = float(info.get("max", BIG_M))

    # Demand constraints over the full horizon.
    for p, info in data["product_grades"].items():
        total_sales = gp.quicksum(sell[p, t] for t in T)
        model.addConstr(total_sales >= float(info.get("demand_min", 0.0)), name=f"demand_min[{p}]")
        demand_max = float(info.get("demand_max", BIG_M))
        if demand_max < 0.999 * BIG_M:
            model.addConstr(total_sales <= demand_max, name=f"demand_max[{p}]")

    for group, info in data.get("demand_groups", {}).items():
        products_in_group = info.get("products", [])
        min_total = float(info.get("base_min_total", 0.0))
        if products_in_group and min_total > 0:
            model.addConstr(
                gp.quicksum(sell[p, t] for p in products_in_group for t in T) >= min_total,
                name=f"demand_group_min[{group}]",
            )

    # Hydrogen consumption identities. Supply limits are not present in the
    # current case data, so these variables are used for reporting/cost extension.
    h2_consumption = {}
    for u, info in data.get("hydrogen_feed_constraints", {}).items():
        if u in units:
            h2_consumption[u] = model.addVars(T, lb=0.0, name=f"hydrogen_consumption[{u}]")
            ratio = float(info.get("ratio", 0.0))
            for t in T:
                model.addConstr(h2_consumption[u][t] == ratio * charge[u, t], name=f"h2_ratio[{u},{t}]")

    # ---- Switching cost variables and constraints ----
    switching_cost_data = data.get("switching_cost", {})
    load_stability_cost_data = data.get("load_stability_cost", {})
    max_load_change_data = data.get("max_load_change", {})

    switch_indicator = model.addVars(units, T, vtype=GRB.BINARY, name="switch")
    delta_F_pos = model.addVars(units, T, lb=0.0, name="delta_F_pos")
    delta_F_neg = model.addVars(units, T, lb=0.0, name="delta_F_neg")

    for u in units:
        modes = data["units"][u].get("modes", ["normal"])
        for t in T:
            if t == T[0]:
                # No switching cost in the first period (no previous state).
                model.addConstr(switch_indicator[u, t] == 0, name=f"switch_init[{u},{t}]")
                model.addConstr(delta_F_pos[u, t] == 0, name=f"delta_F_pos_init[{u},{t}]")
                model.addConstr(delta_F_neg[u, t] == 0, name=f"delta_F_neg_init[{u},{t}]")
                continue
            prev_t = T[T.index(t) - 1]
            # Switch indicator: triggered when on/off status or mode changes.
            model.addConstr(
                switch_indicator[u, t] >= unit_on[u, t] - unit_on[u, prev_t],
                name=f"switch_on[{u},{t}]",
            )
            model.addConstr(
                switch_indicator[u, t] >= unit_on[u, prev_t] - unit_on[u, t],
                name=f"switch_off[{u},{t}]",
            )
            for m in modes:
                model.addConstr(
                    switch_indicator[u, t] >= mode_on[u][m, t] - mode_on[u][m, prev_t],
                    name=f"switch_mode_pos[{u},{m},{t}]",
                )
                model.addConstr(
                    switch_indicator[u, t] >= mode_on[u][m, prev_t] - mode_on[u][m, t],
                    name=f"switch_mode_neg[{u},{m},{t}]",
                )
            # Load stability: penalise |charge[u,t] - charge[u,t-1]| beyond Delta_max.
            delta_max = float(max_load_change_data.get(u, BIG_M))
            model.addConstr(
                delta_F_pos[u, t] >= charge[u, t] - charge[u, prev_t] - delta_max,
                name=f"load_stab_pos[{u},{t}]",
            )
            model.addConstr(
                delta_F_neg[u, t] >= charge[u, prev_t] - charge[u, t] - delta_max,
                name=f"load_stab_neg[{u},{t}]",
            )

    # Objective: revenue minus crude purchase, utilities, inventory, switching,
    # and load-stability costs.
    revenue = gp.quicksum(
        float(data["product_grades"][p]["price"]) * sell[p, t] for p in products for t in T
    )
    crude_cost = gp.quicksum(
        float(data["crude_price_base"][c]) * crude_buy[c, t] for c in crudes for t in T
    )
    utility_cost = gp.LinExpr()
    for u in units:
        coeffs = data.get("unit_utility_coefficients", {}).get(u, [])
        for idx, coeff in enumerate(coeffs):
            if idx < len(utility_names):
                utility_cost += float(coeff) * float(data["utility_price"][utility_names[idx]]) * gp.quicksum(
                    charge[u, t] for t in T
                )
    inventory_cost = gp.quicksum(
        float(data["inventory_nodes"]["crude_tanks"].get("inventory_cost", 0.0))
        * gp.quicksum(crude_inv[c, t] for c in crudes)
        for t in T
    )
    inventory_cost += gp.quicksum(
        float(data["inventory_nodes"][node].get("inventory_cost", 0.0)) * inventory[node, t]
        for node in non_crude_inventory_nodes
        for t in T
    )
    inventory_cost += gp.quicksum(
        float(data["product_pools"][pool].get("inventory_cost", 0.0)) * product_inventory[pool, t]
        for pool in product_pools
        for t in T
    )
    switching_cost = gp.quicksum(
        float(switching_cost_data.get(u, 0.0)) * switch_indicator[u, t]
        for u in units
        for t in T
    )
    load_stability_cost = gp.quicksum(
        float(load_stability_cost_data.get(u, 0.0)) * (delta_F_pos[u, t] + delta_F_neg[u, t])
        for u in units
        for t in T
    )

    model.setObjective(
        revenue - crude_cost - utility_cost - inventory_cost - switching_cost - load_stability_cost,
        GRB.MAXIMIZE,
    )

    variables = {
        "crude_buy": crude_buy,
        "crude_inv": crude_inv,
        "crude_feed": crude_feed,
        "charge": charge,
        "unit_on": unit_on,
        "mode_on": mode_on,
        "mode_charge": mode_charge,
        "cdu_to_naphtha_buffer": cdu_to_naphtha_buffer,
        "cdu_to_distillate_buffer": cdu_to_distillate_buffer,
        "cdu_to_dfhc": cdu_to_dfhc,
        "cdu_to_fcc_feed_buffer": cdu_to_fcc_feed_buffer,
        "cdu_to_fcc": cdu_to_fcc,
        "cdu_to_residual_buffer": cdu_to_residual_buffer,
        "naphtha_to_gasoline_component": naphtha_to_gasoline_component,
        "distillate_to_dfhc": distillate_to_dfhc,
        "distillate_to_dhc": distillate_to_dhc,
        "fcc_buffer_to_fcc": fcc_buffer_to_fcc,
        "residual_to_rohu": residual_to_rohu,
        "hydro_feed_to_dhc": hydro_feed_to_dhc,
        "inventory": inventory,
        "product_inventory": product_inventory,
        "blend": blend,
        "sell": sell,
        "h2_consumption": h2_consumption,
        "crude_quality_mix": None,
        "revenue": revenue,
        "crude_cost": crude_cost,
        "utility_cost": utility_cost,
        "inventory_cost": inventory_cost,
        "switching_cost": switching_cost,
        "load_stability_cost": load_stability_cost,
    }
    return ModelBundle(model=model, variables=variables, sets=sets, warnings=warnings)


def _value(expr: Any) -> float:
    try:
        return float(expr.X)
    except AttributeError:
        return float(expr.getValue())


def _write_csv(path: str, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_solution(bundle: ModelBundle, result_dir: str) -> None:
    """Write compact CSV/JSON solution reports."""
    os.makedirs(result_dir, exist_ok=True)
    model = bundle.model
    v = bundle.variables
    sets = bundle.sets
    data = CASE_DATA

    objective = {
        "status": int(model.Status),
        "objective": model.ObjVal if hasattr(model, "ObjVal") and math.isfinite(model.ObjVal) else None,
        "revenue": _value(v["revenue"]),
        "crude_cost": _value(v["crude_cost"]),
        "utility_cost": _value(v["utility_cost"]),
        "inventory_cost": _value(v["inventory_cost"]),
        "switching_cost": _value(v["switching_cost"]),
        "load_stability_cost": _value(v["load_stability_cost"]),
    }
    with open(os.path.join(result_dir, "objective.json"), "w", encoding="utf-8") as fh:
        json.dump(objective, fh, ensure_ascii=False, indent=2)

    _write_csv(
        os.path.join(result_dir, "unit_schedule.csv"),
        ["period", "unit", "on", "charge", "mode"],
        (
            {
                "period": t + 1,
                "unit": u,
                "on": round(v["unit_on"][u, t].X),
                "charge": v["charge"][u, t].X,
                "mode": ",".join(
                    m
                    for m in data["units"][u].get("modes", ["normal"])
                    if v["mode_on"][u][m, t].X > 0.5
                ),
            }
            for t in sets["T"]
            for u in sets["units"]
        ),
    )

    _write_csv(
        os.path.join(result_dir, "crude_purchase_inventory.csv"),
        ["period", "crude", "purchase", "inventory"],
        (
            {
                "period": t + 1,
                "crude": c,
                "purchase": v["crude_buy"][c, t].X,
                "inventory": v["crude_inv"][c, t].X,
            }
            for t in sets["T"]
            for c in sets["crudes"]
        ),
    )

    _write_csv(
        os.path.join(result_dir, "inventory.csv"),
        ["period", "node", "inventory"],
        (
            {"period": t + 1, "node": node, "inventory": v["inventory"][node, t].X}
            for t in sets["T"]
            for node in sets["non_crude_inventory_nodes"]
        ),
    )

    _write_csv(
        os.path.join(result_dir, "product_inventory.csv"),
        ["period", "pool", "inventory"],
        (
            {"period": t + 1, "pool": pool, "inventory": v["product_inventory"][pool, t].X}
            for t in sets["T"]
            for pool in sets["product_pools"]
        ),
    )

    _write_csv(
        os.path.join(result_dir, "sales.csv"),
        ["period", "product", "sale"],
        (
            {"period": t + 1, "product": p, "sale": v["sell"][p, t].X}
            for t in sets["T"]
            for p in sets["products"]
        ),
    )

    _write_csv(
        os.path.join(result_dir, "blend.csv"),
        ["period", "component_pool", "product_pool", "flow"],
        (
            {
                "period": t + 1,
                "component_pool": cpool,
                "product_pool": ppool,
                "flow": v["blend"][cpool, ppool, t].X,
            }
            for t in sets["T"]
            for cpool in sets["component_pools"]
            for ppool in sets["product_pools"]
            if v["blend"][cpool, ppool, t].X > 1e-7
        ),
    )


def solve(args: argparse.Namespace) -> int:
    errors, warnings = validate_case_data(CASE_DATA)
    for warning in warnings:
        print(f"[warning] {warning}")
    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 2
    if args.validate_only:
        print("Case data validation passed.")
        return 0

    try:
        bundle = build_refinery_model(CASE_DATA, quality_mode=args.quality_mode)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1
    model = bundle.model
    if args.time_limit is not None:
        model.Params.TimeLimit = args.time_limit
    if args.mip_gap is not None:
        model.Params.MIPGap = args.mip_gap
    if args.threads is not None:
        model.Params.Threads = args.threads
    if args.write_lp:
        model.write(args.write_lp)
        print(f"Wrote model file: {args.write_lp}")

    if not args.solve:
        print("Model built successfully. Use --solve to optimize it.")
        return 0

    model.optimize()
    status = model.Status
    gp, GRB = _import_gurobi()
    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        if model.SolCount > 0:
            write_solution(bundle, args.result_dir)
            print(f"Solution written to: {args.result_dir}")
            print(f"Objective value: {model.ObjVal:.6f}")
            return 0
    if status == GRB.INFEASIBLE:
        iis_path = os.path.join(args.result_dir, "infeasible.ilp")
        os.makedirs(args.result_dir, exist_ok=True)
        model.computeIIS()
        model.write(iis_path)
        print(f"Model is infeasible. IIS written to: {iis_path}")
        return 3
    print(f"Optimization ended without a solution. Gurobi status: {status}")
    return 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and solve the refinery scheduling MILP/MIQCP.")
    parser.add_argument("--solve", action="store_true", help="Optimize the model.")
    parser.add_argument("--validate-only", action="store_true", help="Validate data without importing gurobipy.")
    parser.add_argument(
        "--quality-mode",
        choices=["milp", "bilinear-crude"],
        default="milp",
        help="Use aggregate MILP or add bilinear CDU crude quality identities.",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi time limit in seconds.")
    parser.add_argument("--mip-gap", type=float, default=None, help="Relative MIP gap.")
    parser.add_argument("--threads", type=int, default=None, help="Gurobi thread count.")
    parser.add_argument("--write-lp", default=None, help="Optional path for LP/MPS model export.")
    parser.add_argument("--result-dir", default="results", help="Directory for solution CSV/JSON files.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(solve(parse_args()))
