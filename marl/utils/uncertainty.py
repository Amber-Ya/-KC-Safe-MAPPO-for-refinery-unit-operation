"""Uncertainty profiles shared by training and comparison experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


UNCERTAINTY_PROFILES: Dict[str, Dict[str, float | bool]] = {
    "none": {"enabled": False},
    "moderate": {
        "enabled": True,
        "demand_std": 0.12,
        "price_std": 0.08,
        "yield_std": 0.05,
        "crude_price_std": 0.07,
        "unit_outage_prob": 0.03,
        "unit_derate_min": 0.65,
    },
    "stress": {
        "enabled": True,
        "demand_std": 0.20,
        "price_std": 0.12,
        "yield_std": 0.08,
        "crude_price_std": 0.10,
        "unit_outage_prob": 0.06,
        "unit_derate_min": 0.45,
    },
}


def apply_uncertainty_profile(
    env_config: Mapping[str, Any],
    profile: str = "moderate",
) -> Dict[str, Any]:
    """Return a copied env config with the requested uncertainty profile."""
    if profile not in UNCERTAINTY_PROFILES:
        valid = ", ".join(sorted(UNCERTAINTY_PROFILES))
        raise ValueError(f"Unknown uncertainty profile {profile!r}. Valid profiles: {valid}")
    config = deepcopy(dict(env_config))
    config["uncertainty"] = dict(UNCERTAINTY_PROFILES[profile])
    return config
