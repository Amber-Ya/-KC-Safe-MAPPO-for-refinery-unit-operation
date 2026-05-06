from __future__ import annotations

from config import CASE_DATA

import config
from marl.envs.refinery_env import RefinerySchedulingEnv
from marl.utils.config_adapter import ConfigAdapter


def make_env(seed: int = 7) -> RefinerySchedulingEnv:
    return RefinerySchedulingEnv(ConfigAdapter(config).build_env_config(), seed=seed)


def test_reset_returns_seven_agent_observations() -> None:
    env = make_env()
    obs = env.reset(seed=1)
    assert set(obs) == {"CDU1", "CDU2", "DFHC", "FCC1", "FCC2", "ROHU", "DHC"}
    assert len({value.shape for value in obs.values()}) == 1
    assert env.state["t"] == 0


def test_reset_initializes_inventory_and_loads() -> None:
    env = make_env()
    env.reset(seed=1)
    for node, info in CASE_DATA["inventory_nodes"].items():
        assert env.state["inventories"][node] == float(info.get("init", 0.0))
    for unit, load in env.state["unit_loads"].items():
        assert load == 0.0
        assert load <= float(CASE_DATA["units"][unit]["capacity_max"])
