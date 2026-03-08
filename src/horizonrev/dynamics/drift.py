"""Market drift mechanics."""

from __future__ import annotations

import numpy as np

from horizonrev.config import HorizonRevConfig


def initialize_drift_month(state: dict, config: HorizonRevConfig, rng: np.random.Generator) -> None:
    if not config.drift_enabled:
        state["_drift_month_active"] = None
        return
    if config.drift_randomize:
        low, high = config.drift_month_range
        state["_drift_month_active"] = int(rng.integers(low, high + 1))
    else:
        state["_drift_month_active"] = config.drift_month


def apply_market_drift_if_needed(state: dict, config: HorizonRevConfig) -> bool:
    if not config.drift_enabled:
        return False
    drift_month = state.get("_drift_month_active", config.drift_month)
    if drift_month is None or state["month"] != drift_month:
        return False
    if state.get("_drift_applied", False):
        return False

    strength = config.drift_strength
    state["discount_effect_smb"] *= 1.0 + (config.drift_discount_multiplier_smb - 1.0) * strength
    state["onboarding_effect_ent"] *= 1.0 + (config.drift_onboarding_multiplier_ent - 1.0) * strength
    state["smb_demand"] = max(0.2, state["smb_demand"] + config.drift_demand_shift_smb * strength)
    state["ent_demand"] = max(0.2, state["ent_demand"] + config.drift_demand_shift_ent * strength)
    state["_drift_applied"] = True
    return True
