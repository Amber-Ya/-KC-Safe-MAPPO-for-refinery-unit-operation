from __future__ import annotations

import config
from marl.envs.refinery_env import RefinerySchedulingEnv
from marl.utils.config_adapter import ConfigAdapter


def test_inventory_bounds_after_step() -> None:
    env = RefinerySchedulingEnv(ConfigAdapter(config).build_env_config(), seed=5)
    env.reset(seed=5)
    _, _, _, info = env.step({agent: 2 for agent in env.agents})
    for node, amount in info["inventories"].items():
        spec = config.CASE_DATA["inventory_nodes"][node]
        assert amount >= spec["min"] - 1e-6
        assert amount <= spec["max"] + 1e-6
    for pool, amount in info["product_pools"].items():
        spec = config.CASE_DATA["product_pools"][pool]
        assert amount >= spec["min"] - 1e-6
        assert amount <= spec["max"] + 1e-6


def test_component_blending_balance_for_one_step() -> None:
    env = RefinerySchedulingEnv(ConfigAdapter(config).build_env_config(), seed=6)
    env.reset(seed=6)
    old_gasoline_component = env.state["inventories"]["gasoline_component_pool"]
    _, _, _, info = env.step({"CDU1": 2, "CDU2": 1, "DFHC": 1, "FCC1": 1, "FCC2": 1, "ROHU": 1, "DHC": 1})
    inflow = sum(v for (src, dst), v in info["flows"].items() if dst == "gasoline_component_pool")
    outflow = sum(v for (src, dst), v in info["blend_flows"].items() if src == "gasoline_component_pool")
    new_gasoline_component = info["inventories"]["gasoline_component_pool"]
    assert abs(new_gasoline_component - (old_gasoline_component + inflow - outflow)) <= 1e-6


def test_product_grade_sales_respect_real_caps_not_big_m() -> None:
    env = RefinerySchedulingEnv(ConfigAdapter(config).build_env_config(), seed=7)
    env.reset(seed=7)
    done = False
    while not done:
        _, _, dones, _ = env.step({agent: 6 for agent in env.agents})
        done = dones["__all__"]

    sales = env.state["cumulative_product_sales"]
    assert sales["W95"] <= config.CASE_DATA["product_grades"]["W95"]["demand_max"] + 1e-6
    assert sales["STR"] <= config.CASE_DATA["product_grades"]["STR"]["demand_max"] + 1e-6
    assert sales["W92"] > config.BIG_M


def test_reward_revenue_uses_grade_level_sales_prices() -> None:
    env = RefinerySchedulingEnv(ConfigAdapter(config).build_env_config(), seed=8)
    env.reset(seed=8)
    _, _, _, info = env.step({agent: 6 for agent in env.agents})
    expected_revenue = sum(
        amount * config.CASE_DATA["product_grades"][product]["price"]
        for product, amount in info["product_sales"].items()
    )
    assert abs(info["revenue"] - expected_revenue) <= 1e-6
