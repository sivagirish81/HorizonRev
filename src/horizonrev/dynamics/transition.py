"""Core transition mechanics for HorizonRev."""

from __future__ import annotations

from typing import Any, Dict

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
    cohort_len = config.max_cohort_age + 1
    smb_counts = np.zeros(cohort_len, dtype=np.float64)
    ent_counts = np.zeros(cohort_len, dtype=np.float64)
    smb_quality = np.zeros(cohort_len, dtype=np.float64)
    ent_quality = np.zeros(cohort_len, dtype=np.float64)
    smb_promo = np.zeros(cohort_len, dtype=np.float64)
    ent_promo = np.zeros(cohort_len, dtype=np.float64)

    smb_share = 0.62
    smb_total = config.initial_customer_base * smb_share
    ent_total = config.initial_customer_base * (1.0 - smb_share)
    smb_counts[:6] = np.asarray([0.17, 0.15, 0.13, 0.12, 0.11, 0.1]) * smb_total
    ent_counts[:6] = np.asarray([0.13, 0.12, 0.1, 0.1, 0.1, 0.1]) * ent_total
    smb_counts[-1] += max(0.0, smb_total - float(np.sum(smb_counts)))
    ent_counts[-1] += max(0.0, ent_total - float(np.sum(ent_counts)))
    smb_quality[:] = config.quality_mu_smb
    ent_quality[:] = config.quality_mu_ent

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
        "last_churn_volatility": 0.0,
        "prev_churn": 0.0,
        "pct_young_customers": 0.0,
        "avg_quality": (config.quality_mu_smb + config.quality_mu_ent) / 2.0,
        "active_shocks": {},
        "promo_expiry_timer": 0,
        "promo_expiry_intensity": 0.0,
        "cohorts": {"SMB": smb_counts, "ENT": ent_counts},
        "quality": {"SMB": smb_quality, "ENT": ent_quality},
        "promo_tag": {"SMB": smb_promo, "ENT": ent_promo},
        "total_customers": float(np.sum(smb_counts) + np.sum(ent_counts)),
        "new_customers": 0.0,
        "churned_customers": 0.0,
        "_drift_applied": False,
        "_drift_month_active": None,
    }


def apply_action(state: dict[str, Any], action: int, config: HorizonRevConfig, delayed_queue) -> None:
    state["onboarding_invest_active"] = False
    prev_discount = state["discount_level"]

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
    if prev_discount >= config.promo_discount_threshold and state["discount_level"] < prev_discount:
        state["promo_expiry_timer"] = config.promo_expiry_duration
        state["promo_expiry_intensity"] = prev_discount - state["discount_level"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _base_hazard(segment: str, age: int, config: HorizonRevConfig) -> float:
    if segment == "SMB":
        raw = _sigmoid(config.theta0_smb + config.theta_age_smb * np.log1p(age))
    else:
        raw = _sigmoid(config.theta0_ent + config.theta_age_ent * np.log1p(age))
    return float(np.clip(raw, 0.0, config.hazard_max_base))


def _weighted_mean(existing_count: float, existing_val: float, incoming_count: float, incoming_val: float) -> float:
    total = existing_count + incoming_count
    if total <= 1e-9:
        return 0.0
    return (existing_count * existing_val + incoming_count * incoming_val) / total


def _initialize_new_quality(segment: str, state: dict[str, Any], config: HorizonRevConfig, rng: np.random.Generator) -> float:
    discount = state["discount_level"]
    onboarding_boost = config.onboarding_quality_boost if state["onboarding_invest_active"] else 0.0
    if segment == "SMB":
        mean = config.quality_mu_smb - config.discount_quality_sensitivity_smb * discount + onboarding_boost
        sampled = rng.normal(mean, config.quality_sigma_smb)
    else:
        mean = config.quality_mu_ent - config.discount_quality_sensitivity_ent * discount + onboarding_boost * 1.2
        sampled = rng.normal(mean, config.quality_sigma_ent)
    return float(np.clip(sampled, config.quality_min, config.quality_max))


def _shock_modifiers(state: dict[str, Any], config: HorizonRevConfig) -> Dict[str, float]:
    conv_mult_smb = 1.0
    conv_mult_ent = 1.0
    demand_mult_smb = 1.0
    demand_mult_ent = 1.0
    churn_add_smb = 0.0
    churn_add_ent = 0.0

    for shock_name in state["active_shocks"].keys():
        if shock_name == "competitor_entry":
            demand_mult_smb *= config.shock_competitor_demand_mult_smb
            demand_mult_ent *= config.shock_competitor_demand_mult_ent
            churn_add_smb += config.shock_competitor_churn_bump_smb
            churn_add_ent += config.shock_competitor_churn_bump_ent
        elif shock_name == "budget_freeze":
            conv_mult_ent *= config.shock_budget_conv_mult_ent
            churn_add_ent += config.shock_budget_churn_bump_ent
        elif shock_name == "outage_or_trust_event":
            churn_add_ent += config.shock_outage_churn_bump_ent
        elif shock_name == "macro_downturn":
            demand_mult_smb *= config.shock_macro_demand_mult_smb
            demand_mult_ent *= config.shock_macro_demand_mult_ent
            churn_add_smb += config.shock_macro_churn_bump_all
            churn_add_ent += config.shock_macro_churn_bump_all

    return {
        "conv_mult_smb": conv_mult_smb,
        "conv_mult_ent": conv_mult_ent,
        "demand_mult_smb": demand_mult_smb,
        "demand_mult_ent": demand_mult_ent,
        "churn_add_smb": churn_add_smb,
        "churn_add_ent": churn_add_ent,
    }


def sample_and_update_shocks(state: dict[str, Any], config: HorizonRevConfig, rng: np.random.Generator) -> list[str]:
    active = state["active_shocks"]
    for name in list(active.keys()):
        active[name]["remaining"] -= 1
        if active[name]["remaining"] <= 0:
            del active[name]

    events = [
        ("competitor_entry", config.shock_competitor_entry_prob, config.shock_competitor_duration),
        ("budget_freeze", config.shock_budget_freeze_prob, config.shock_budget_duration),
        ("outage_or_trust_event", config.shock_outage_prob, config.shock_outage_duration),
        ("macro_downturn", config.shock_macro_prob, config.shock_macro_duration),
    ]
    triggered = []
    for name, prob, duration in events:
        if name not in active and float(rng.random()) < prob:
            active[name] = {"remaining": int(duration)}
            triggered.append(name)

    if state["promo_expiry_timer"] > 0:
        state["promo_expiry_timer"] -= 1
        if state["promo_expiry_timer"] == 0:
            state["promo_expiry_intensity"] = 0.0
    return triggered


def compute_segment_metrics(
    state: dict[str, Any],
    config: HorizonRevConfig,
    delayed_churn_delta: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    focus_smb = 0.012 if state["sales_focus"] == 0 else -0.004
    focus_ent = 0.012 if state["sales_focus"] == 1 else -0.004
    pricing_lift = config.pricing_ab_lift if state["pricing_test_active"] else 0.0
    shock = _shock_modifiers(state, config)

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
    conv_smb *= shock["conv_mult_smb"]
    conv_ent *= shock["conv_mult_ent"]
    conv_smb = float(np.clip(conv_smb * state["smb_demand"] * shock["demand_mult_smb"], 0.0, 1.0))
    conv_ent = float(np.clip(conv_ent * state["ent_demand"] * shock["demand_mult_ent"], 0.0, 1.0))

    return {
        "SMB": {"conversion": conv_smb, "demand": state["smb_demand"], "shock": shock},
        "ENT": {"conversion": conv_ent, "demand": state["ent_demand"], "shock": shock},
    }


def update_arr_and_base(
    state: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    delayed_churn_delta: float,
    config: HorizonRevConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float, dict[str, Any]]:
    prev_arr = state["arr"]
    cohorts = state["cohorts"]
    quality = state["quality"]
    promo = state["promo_tag"]
    cohort_len = config.max_cohort_age + 1

    new_customers = {}
    new_quality = {}
    new_promo = {}
    for seg in ("SMB", "ENT"):
        traffic = config.traffic_smb if seg == "SMB" else config.traffic_ent
        conv = metrics[seg]["conversion"]
        acquired = max(0.0, conv * traffic)
        new_customers[seg] = acquired
        q_val = _initialize_new_quality(seg, state, config, rng)
        new_quality[seg] = q_val
        high_discount = max(0.0, state["discount_level"] - config.promo_discount_threshold)
        new_promo[seg] = high_discount * config.promo_tag_scale

    churned_by_seg = {"SMB": 0.0, "ENT": 0.0}
    churn_rate_by_seg = {"SMB": 0.0, "ENT": 0.0}
    total_before = 0.0
    total_after = 0.0
    weighted_quality = 0.0

    for seg in ("SMB", "ENT"):
        seg_counts = cohorts[seg]
        seg_quality = quality[seg]
        seg_promo = promo[seg]
        prev_total = float(np.sum(seg_counts))
        total_before += prev_total
        survivors = np.zeros_like(seg_counts)

        for age in range(cohort_len):
            base = _base_hazard(seg, age, config)
            age_weight = float(np.exp(-age / config.onboarding_young_tau))
            onboarding_effect = 0.0
            if state["onboarding_invest_active"]:
                base_effect = config.onboarding_effect_smb if seg == "SMB" else config.onboarding_effect_ent
                onboarding_effect = -base_effect * age_weight * state["onboarding_effect_ent"]
            experiment_effect = config.experiment_churn_effect_smb if (seg == "SMB" and state["pricing_test_active"]) else 0.0
            experiment_effect += config.experiment_churn_effect_ent if (seg == "ENT" and state["pricing_test_active"]) else 0.0

            shock_add = metrics[seg]["shock"]["churn_add_smb"] if seg == "SMB" else metrics[seg]["shock"]["churn_add_ent"]
            trust_amp = config.drift_shock_sensitivity_ent if seg == "ENT" else 1.0
            promo_spike = (
                state["promo_expiry_intensity"]
                * config.promo_expiry_multiplier
                * float(np.exp(-age / config.promo_age_tau))
                * seg_promo[age]
            )
            noise = float(rng.normal(0.0, config.noise_churn_std))
            hazard = base + delayed_churn_delta + onboarding_effect + experiment_effect + shock_add * trust_amp + promo_spike + noise
            beta_q = config.quality_beta_smb if seg == "SMB" else config.quality_beta_ent
            hazard *= float(np.exp(-beta_q * seg_quality[age]))
            hazard = float(np.clip(hazard, 0.0, config.hazard_max_final))

            churned = seg_counts[age] * hazard
            survivors[age] = max(0.0, seg_counts[age] - churned)
            churned_by_seg[seg] += churned

        next_counts = np.zeros_like(seg_counts)
        next_quality = np.zeros_like(seg_quality)
        next_promo = np.zeros_like(seg_promo)
        for age in range(cohort_len - 1, 0, -1):
            if age == cohort_len - 1:
                incoming_count = survivors[age] + survivors[age - 1]
                incoming_quality = _weighted_mean(survivors[age], seg_quality[age], survivors[age - 1], seg_quality[age - 1])
                incoming_promo = _weighted_mean(survivors[age], seg_promo[age], survivors[age - 1], seg_promo[age - 1])
            else:
                incoming_count = survivors[age - 1]
                incoming_quality = seg_quality[age - 1]
                incoming_promo = seg_promo[age - 1] * 0.86
            next_counts[age] = incoming_count
            next_quality[age] = incoming_quality
            next_promo[age] = incoming_promo

        next_counts[0] = new_customers[seg]
        next_quality[0] = new_quality[seg]
        next_promo[0] = new_promo[seg]
        cohorts[seg] = next_counts
        quality[seg] = next_quality
        promo[seg] = next_promo

        seg_after = float(np.sum(next_counts))
        total_after += seg_after
        if prev_total > 1e-9:
            churn_rate_by_seg[seg] = churned_by_seg[seg] / prev_total
        weighted_quality += float(np.sum(next_counts * next_quality))

    conv_total = 0.6 * metrics["SMB"]["conversion"] + 0.4 * metrics["ENT"]["conversion"]
    churn_total = (churned_by_seg["SMB"] + churned_by_seg["ENT"]) / max(1e-9, total_before)
    arr_delta = (
        (new_customers["SMB"] - churned_by_seg["SMB"]) * config.arpa_smb
        + (new_customers["ENT"] - churned_by_seg["ENT"]) * config.arpa_ent
    )
    state["arr"] = max(5_000.0, state["arr"] + arr_delta)
    state["last_conversion"] = conv_total
    state["last_churn"] = churn_total
    state["last_churn_volatility"] = abs(churn_total - state["prev_churn"])
    state["prev_churn"] = churn_total
    state["customer_base"] = total_after
    state["total_customers"] = total_after
    state["new_customers"] = new_customers["SMB"] + new_customers["ENT"]
    state["churned_customers"] = churned_by_seg["SMB"] + churned_by_seg["ENT"]
    young = float(np.sum(cohorts["SMB"][:3]) + np.sum(cohorts["ENT"][:3]))
    state["pct_young_customers"] = young / max(1e-9, total_after)
    state["avg_quality"] = weighted_quality / max(1e-9, total_after)

    cohort_summary = {
        "age_0_2": float(np.sum(cohorts["SMB"][:3]) + np.sum(cohorts["ENT"][:3])),
        "age_3_5": float(np.sum(cohorts["SMB"][3:6]) + np.sum(cohorts["ENT"][3:6])),
        "age_6_plus": float(np.sum(cohorts["SMB"][6:]) + np.sum(cohorts["ENT"][6:])),
    }
    segment_metrics = {
        "SMB": {
            "conversion": float(metrics["SMB"]["conversion"]),
            "churn": float(churn_rate_by_seg["SMB"]),
            "new_customers": float(new_customers["SMB"]),
            "churned_customers": float(churned_by_seg["SMB"]),
        },
        "ENT": {
            "conversion": float(metrics["ENT"]["conversion"]),
            "churn": float(churn_rate_by_seg["ENT"]),
            "new_customers": float(new_customers["ENT"]),
            "churned_customers": float(churned_by_seg["ENT"]),
        },
    }
    details = {"segment_metrics": segment_metrics, "cohort_summary": cohort_summary}
    return prev_arr, conv_total, churn_total, details
