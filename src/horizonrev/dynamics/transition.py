"""Core transition mechanics for HorizonRev."""

from __future__ import annotations

from typing import Any

import numpy as np

from horizonrev.config import HorizonRevConfig
from horizonrev.dynamics.delayed import schedule_effect
from horizonrev.dynamics.experiments import (
    DECREASE_DISCOUNT,
    INCREASE_DISCOUNT,
    INVEST_ONBOARDING,
    LAUNCH_PRICING_AB,
    SHIFT_SALES_ENT,
    SHIFT_SALES_SMB,
    STOP_PRICING_AB,
)


def create_initial_state(config: HorizonRevConfig) -> dict[str, Any]:
    return {
        "month": 1,
        "arr": float(config.initial_arr),
        "customer_base": float(config.initial_customer_base),
        "discount_level": 0.2,
        "sales_focus": 0,  # 0 SMB, 1 ENT
        "pricing_test_active": False,
        "onboarding_invest_active": False,
        "pricing_test_months": 0,
        "discount_effect_smb": float(config.discount_effect_smb),
        "discount_effect_ent": float(config.discount_effect_ent),
        "onboarding_effect_ent": 1.0,
        "smb_demand": float(config.demand_smb_init),
        "ent_demand": float(config.demand_ent_init),
        "last_conversion": 0.0,
        "last_churn": 0.0,
        "_drift_applied": False,
    }


def apply_action(state: dict[str, Any], action: int, config: HorizonRevConfig, delayed_queue) -> None:
    state["onboarding_invest_active"] = False

    if action == INCREASE_DISCOUNT:
        state["discount_level"] = min(1.0, state["discount_level"] + config.discount_step)
        target_month = state["month"] + config.delay_months
        if target_month <= config.episode_length:
            schedule_effect(
                delayed_queue,
                target_month,
                config.delayed_discount_churn_delta,
                "discount_backfire",
            )
    elif action == DECREASE_DISCOUNT:
        state["discount_level"] = max(0.0, state["discount_level"] - config.discount_step)
    elif action == LAUNCH_PRICING_AB:
        state["pricing_test_active"] = True
    elif action == STOP_PRICING_AB:
        state["pricing_test_active"] = False
    elif action == INVEST_ONBOARDING:
        state["onboarding_invest_active"] = True
        target_month = state["month"] + config.delay_months
        if target_month <= config.episode_length:
            trust_multiplier = state["onboarding_effect_ent"]
            schedule_effect(
                delayed_queue,
                target_month,
                config.delayed_onboarding_churn_delta * trust_multiplier,
                "onboarding_payoff",
            )
    elif action == SHIFT_SALES_SMB:
        state["sales_focus"] = 0
    elif action == SHIFT_SALES_ENT:
        state["sales_focus"] = 1

    if state["pricing_test_active"]:
        state["pricing_test_months"] += 1


def compute_segment_metrics(
    state: dict[str, Any],
    config: HorizonRevConfig,
    delayed_churn_delta: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    focus_smb = 0.012 if state["sales_focus"] == 0 else -0.004
    focus_ent = 0.012 if state["sales_focus"] == 1 else -0.004
    pricing_lift = config.pricing_ab_lift if state["pricing_test_active"] else 0.0

    conv_smb = (
        config.base_conv_smb
        + state["discount_level"] * state["discount_effect_smb"]
        + pricing_lift
        + focus_smb
        + rng.normal(0.0, config.noise_conv_std)
    )
    conv_ent = (
        config.base_conv_ent
        + state["discount_level"] * state["discount_effect_ent"]
        + pricing_lift
        + focus_ent
        + rng.normal(0.0, config.noise_conv_std)
    )
    conv_smb = float(np.clip(conv_smb * state["smb_demand"], 0.0, 1.0))
    conv_ent = float(np.clip(conv_ent * state["ent_demand"], 0.0, 1.0))

    onboarding_local_relief = -0.004 if state["onboarding_invest_active"] else 0.0
    churn_smb = config.base_churn_smb + delayed_churn_delta + onboarding_local_relief
    churn_ent = config.base_churn_ent + delayed_churn_delta + onboarding_local_relief * state["onboarding_effect_ent"]
    churn_smb += rng.normal(0.0, config.noise_churn_std)
    churn_ent += rng.normal(0.0, config.noise_churn_std)
    churn_smb = float(np.clip(churn_smb, 0.0, 0.25))
    churn_ent = float(np.clip(churn_ent, 0.0, 0.25))

    return {
        "SMB": {"conversion": conv_smb, "churn": churn_smb, "demand": state["smb_demand"]},
        "ENT": {"conversion": conv_ent, "churn": churn_ent, "demand": state["ent_demand"]},
    }


def update_arr_and_base(
    state: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    config: HorizonRevConfig,
) -> tuple[float, float, float]:
    prev_arr = state["arr"]
    conv_total = 0.6 * metrics["SMB"]["conversion"] + 0.4 * metrics["ENT"]["conversion"]
    churn_total = 0.55 * metrics["SMB"]["churn"] + 0.45 * metrics["ENT"]["churn"]

    new_customers = conv_total * config.traffic_scale
    churned_customers = churn_total * state["customer_base"] * 0.16
    state["customer_base"] = max(10.0, state["customer_base"] + new_customers - churned_customers)

    arr_delta = (new_customers - churned_customers) * config.arpa
    state["arr"] = max(5_000.0, state["arr"] + arr_delta)
    state["last_conversion"] = conv_total
    state["last_churn"] = churn_total
    return prev_arr, conv_total, churn_total
