from __future__ import annotations

from horizonrev import HorizonRevConfig, HorizonRevEnv
from horizonrev.reward import score_report


GOOD_REPORT = """
Hypothesis: Discount plus onboarding improves ARR over six months.
Action: Increase discount slightly and invest in onboarding now.
Expected Impact: conversion rises and churn falls after delayed payoff.
Risks: delayed churn backfire if discount is overused during drift.
Next Step: Reassess month, ARR, churn, conversion, and discount before next action.
"""


def test_structured_report_scores_higher():
    q_good, tok_good, _ = score_report(
        GOOD_REPORT,
        required_markers=HorizonRevConfig.default().report_required_markers,
        required_metrics=HorizonRevConfig.default().report_required_metrics,
        word_cap=HorizonRevConfig.default().report_word_cap,
    )
    q_bad, tok_bad, _ = score_report(
        "This is generic filler text with no clear sections and no direct strategy.",
        required_markers=HorizonRevConfig.default().report_required_markers,
        required_metrics=HorizonRevConfig.default().report_required_metrics,
        word_cap=HorizonRevConfig.default().report_word_cap,
    )
    assert tok_good > 0 and tok_bad > 0
    assert q_good > q_bad


def test_long_low_quality_text_gets_no_token_bonus():
    cfg = HorizonRevConfig.default()
    env = HorizonRevEnv(cfg)
    env.reset(seed=1)
    low_quality_long = "word " * 2500
    _, _, _, info = env.step(0, agent_report=low_quality_long)
    rc = info["reward_components"]
    assert rc["token_count"] == cfg.report_word_cap
    assert rc["planning_quality_score"] < cfg.quality_gate_threshold
    assert rc["token_bonus"] == 0.0


def _run_with_reports(reward_mode: str, report: str) -> float:
    cfg = HorizonRevConfig(**{**HorizonRevConfig.default().__dict__, "reward_mode": reward_mode})
    env = HorizonRevEnv(cfg)
    env.reset(seed=7)
    total = 0.0
    for action in [1, 3, 4, 6, 2, 7]:
        _, reward, done, _ = env.step(action, agent_report=report)
        total += reward
        if done:
            break
    return total


def test_uncapped_mode_can_exceed_capped_mode_same_rollout():
    capped_total = _run_with_reports("capped", GOOD_REPORT)
    uncapped_total = _run_with_reports("uncapped", GOOD_REPORT)
    assert uncapped_total > capped_total
