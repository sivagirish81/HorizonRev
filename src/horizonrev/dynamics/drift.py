"""Market drift mechanics."""

from __future__ import annotations

from horizonrev.config import HorizonRevConfig


def apply_market_drift_if_needed(state: dict, config: HorizonRevConfig) -> bool:
    if not config.drift_enabled:
        return False
    if state["month"] != config.drift_month:
        return False
    if state.get("_drift_applied", False):
        return False

    state["discount_effect_smb"] *= config.drift_discount_multiplier_smb
    state["onboarding_effect_ent"] *= config.drift_onboarding_multiplier_ent
    state["smb_demand"] = max(0.2, state["smb_demand"] + config.drift_demand_shift_smb)
    state["ent_demand"] = max(0.2, state["ent_demand"] + config.drift_demand_shift_ent)
    state["_drift_applied"] = True
    return True
