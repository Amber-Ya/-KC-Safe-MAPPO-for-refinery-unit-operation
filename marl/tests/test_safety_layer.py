from __future__ import annotations

import config
from marl.envs.refinery_env import RefinerySchedulingEnv
from marl.utils.config_adapter import ConfigAdapter


def make_env() -> RefinerySchedulingEnv:
    env = RefinerySchedulingEnv(ConfigAdapter(config).build_env_config(), seed=11)
    env.reset(seed=11)
    return env


def test_upstream_shortage_masks_increase_for_secondary() -> None:
    env = make_env()
    env.state["inventories"]["distillate_buffer"] = 0.0
    mask = env.safety_layer.get_action_mask(env.state, "DFHC")
    assert mask[2] == 0.0


def test_downstream_full_masks_increase() -> None:
    env = make_env()
    env.state["inventories"]["gasoline_component_pool"] = config.CASE_DATA["inventory_nodes"]["gasoline_component_pool"]["max"]
    env.state["inventories"]["diesel_jet_component_pool"] = config.CASE_DATA["inventory_nodes"]["diesel_jet_component_pool"]["max"]
    env.state["inventories"]["lpg_component_pool"] = config.CASE_DATA["inventory_nodes"]["lpg_component_pool"]["max"]
    mask = env.safety_layer.get_action_mask(env.state, "FCC1")
    assert mask[2] == 0.0


def test_repair_load_respects_capacity() -> None:
    env = make_env()
    load = env.safety_layer.repair_load("CDU1", 1e6, env.state)
    assert load <= config.CASE_DATA["units"]["CDU1"]["capacity_max"]


def test_repaired_action_does_not_make_obvious_inventory_violation() -> None:
    env = make_env()
    env.state["inventories"]["fcc_feed_buffer"] = 0.0
    safe = env.safety_layer.repair_joint_action(env.state, {"FCC1": 2})
    assert safe["FCC1"] != 2
