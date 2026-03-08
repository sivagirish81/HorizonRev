"""Action constants and simple policies."""

from __future__ import annotations

from typing import Dict

from horizonrev.config import HorizonRevConfig

NOOP = 0
INCREASE_DISCOUNT = 1
DECREASE_DISCOUNT = 2
LAUNCH_PRICING_AB = 3
INVEST_ONBOARDING = 4
SHIFT_SALES_SMB = 5
SHIFT_SALES_ENT = 6
STOP_PRICING_AB = 7

ACTION_NAMES: Dict[int, str] = {
    NOOP: "NOOP",
    INCREASE_DISCOUNT: "INCREASE_DISCOUNT",
    DECREASE_DISCOUNT: "DECREASE_DISCOUNT",
    LAUNCH_PRICING_AB: "LAUNCH_PRICING_AB",
    INVEST_ONBOARDING: "INVEST_ONBOARDING",
    SHIFT_SALES_SMB: "SHIFT_SALES_SMB",
    SHIFT_SALES_ENT: "SHIFT_SALES_ENT",
    STOP_PRICING_AB: "STOP_PRICING_AB",
}


def heuristic_action(obs) -> int:
    """A compact policy to demonstrate long-horizon planning."""
    month = int(round(float(obs[0]) * HorizonRevConfig.default().episode_length))
    churn = float(obs[3])
    discount = float(obs[4])
    pricing_active = bool(round(float(obs[7])))

    if month <= 3:
        if discount < 0.4:
            return INCREASE_DISCOUNT
        return LAUNCH_PRICING_AB if not pricing_active else SHIFT_SALES_SMB

    if month == 6:
        return INVEST_ONBOARDING

    if churn > 0.38:
        return INVEST_ONBOARDING
    if pricing_active and month >= 10:
        return STOP_PRICING_AB
    if month >= 9 and discount > 0.3:
        return DECREASE_DISCOUNT
    return SHIFT_SALES_ENT
