"""Reward computation."""

from __future__ import annotations

import math
import re
from typing import Iterable

from horizonrev.config import HorizonRevConfig


def compute_confounding_penalty(pricing_test_active: bool, onboarding_invest_active: bool, pricing_test_months: int, config: HorizonRevConfig) -> float:
    penalty = 0.0
    if pricing_test_active and onboarding_invest_active:
        penalty += config.confounding_penalty_if_dual
    if pricing_test_months > config.long_pricing_test_months_threshold:
        penalty += config.long_pricing_test_penalty
    return penalty


def score_report(
    text: str,
    required_markers: Iterable[str],
    required_metrics: Iterable[str],
    word_cap: int,
) -> tuple[float, int, float]:
    clean_text = (text or "").strip()
    words = clean_text.split()
    token_count = min(len(words), int(word_cap))
    truncated_text = " ".join(words[:token_count])
    lowered = truncated_text.lower()

    markers = list(required_markers)
    marker_hits = sum(1 for marker in markers if marker.lower() in lowered)
    marker_score = marker_hits / max(1, len(markers))

    metric_hits = 0
    for metric in required_metrics:
        pattern = r"\b" + re.escape(metric.lower()) + r"\b"
        if re.search(pattern, lowered):
            metric_hits += 1
    metrics_score = 1.0 if metric_hits >= 2 else metric_hits / 2.0

    quality_score = (marker_score + metrics_score) / 2.0
    token_bonus_raw = math.log1p(float(token_count))
    return float(quality_score), int(token_count), float(token_bonus_raw)


def compute_reward(
    prev_arr: float,
    new_arr: float,
    churn: float,
    churn_volatility: float,
    confounding_penalty: float,
    agent_report: str,
    is_terminal: bool,
    config: HorizonRevConfig,
) -> tuple[float, dict]:
    delta_arr_component = (new_arr - prev_arr) / config.reward_arr_scale
    churn_penalty = config.reward_alpha * churn
    conf_penalty = config.reward_beta * confounding_penalty
    base_reward = delta_arr_component - churn_penalty - conf_penalty

    quality_score, token_count, token_bonus_raw = score_report(
        text=agent_report,
        required_markers=config.report_required_markers,
        required_metrics=config.report_required_metrics,
        word_cap=config.report_word_cap,
    )

    token_bonus = quality_score * token_bonus_raw
    if quality_score < config.quality_gate_threshold:
        token_bonus = 0.0
    if config.reward_mode == "capped":
        token_bonus = min(token_bonus, config.token_bonus_cap_capped)
    token_bonus_component = config.lambda_token * token_bonus

    low_quality_verbosity_penalty = 0.0
    if quality_score < config.quality_gate_threshold and token_count > config.low_quality_token_threshold:
        low_quality_verbosity_penalty = config.low_quality_verbosity_penalty

    churn_volatility_penalty = config.churn_volatility_penalty_weight * churn_volatility
    reward = base_reward + token_bonus_component - low_quality_verbosity_penalty - churn_volatility_penalty

    if is_terminal and new_arr >= config.terminal_arr_threshold and churn <= config.terminal_churn_threshold:
        reward += config.terminal_bonus

    return float(reward), {
        "delta_arr_component": float(delta_arr_component),
        "churn_penalty": float(churn_penalty),
        "confounding_penalty": float(conf_penalty),
        "planning_quality_score": float(quality_score),
        "token_count": int(token_count),
        "token_bonus": float(token_bonus),
        "token_bonus_component": float(token_bonus_component),
        "low_quality_verbosity_penalty": float(low_quality_verbosity_penalty),
        "churn_volatility_penalty": float(churn_volatility_penalty),
        "base_reward": float(base_reward),
        "delta_arr": float(delta_arr_component),
    }
