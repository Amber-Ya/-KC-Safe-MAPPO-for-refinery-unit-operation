from __future__ import annotations

from config import CASE_DATA

import config
from marl.envs.refinery_env import RefinerySchedulingEnv
from marl.utils.config_adapter import ConfigAdapter
from marl.utils.uncertainty import apply_uncertainty_profile


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


def test_uncertain_reset_samples_multipliers_and_step_profit() -> None:
    env_config = apply_uncertainty_profile(ConfigAdapter(config).build_env_config(), "moderate")
    env = RefinerySchedulingEnv(env_config, seed=13)
    env.reset(seed=13)
    demand_multipliers = env.state["demand_multipliers"]
    assert any(abs(value - 1.0) > 1e-8 for value in demand_multipliers.values())
    assert all(0.0 < value <= 1.0 for value in env.state["unit_availability"].values())
    _, _, _, info = env.step({agent: 1 for agent in env.agents})
    assert "profit" in info
