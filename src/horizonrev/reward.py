"""Reward computation."""

from __future__ import annotations

from horizonrev.config import HorizonRevConfig


def compute_confounding_penalty(pricing_test_active: bool, onboarding_invest_active: bool, pricing_test_months: int, config: HorizonRevConfig) -> float:
    penalty = 0.0
    if pricing_test_active and onboarding_invest_active:
        penalty += config.confounding_penalty_if_dual
    if pricing_test_months > config.long_pricing_test_months_threshold:
        penalty += config.long_pricing_test_penalty
    return penalty


def compute_reward(
    prev_arr: float,
    new_arr: float,
    churn: float,
    confounding_penalty: float,
    is_terminal: bool,
    config: HorizonRevConfig,
) -> tuple[float, dict]:
    delta_arr = (new_arr - prev_arr) / config.reward_arr_scale
    churn_penalty = config.reward_alpha * churn
    conf_penalty = config.reward_beta * confounding_penalty
    reward = delta_arr - churn_penalty - conf_penalty

    if is_terminal and new_arr >= config.terminal_arr_threshold and churn <= config.terminal_churn_threshold:
        reward += config.terminal_bonus

    return float(reward), {
        "delta_arr": float(delta_arr),
        "churn_penalty": float(churn_penalty),
        "confounding_penalty": float(conf_penalty),
    }
