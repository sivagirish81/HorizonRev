from __future__ import annotations

import numpy as np

from horizonrev import HorizonRevConfig, HorizonRevEnv, run_monte_carlo
from horizonrev.dynamics.experiments import heuristic_action


def test_cohort_conservation_and_non_negative():
    env = HorizonRevEnv(HorizonRevConfig.base_case())
    env.reset(seed=21)
    for action in [1, 1, 4, 2, 6, 7]:
        _, _, done, _ = env.step(action)
        for seg in ("SMB", "ENT"):
            counts = env.state["cohorts"][seg]
            assert np.all(counts >= 0.0)
        if done:
            break
    total_from_cohorts = float(np.sum(env.state["cohorts"]["SMB"]) + np.sum(env.state["cohorts"]["ENT"]))
    assert abs(total_from_cohorts - env.state["total_customers"]) < 1e-6


def test_promo_expiry_effect_after_discount_drop():
    env = HorizonRevEnv(HorizonRevConfig.base_case())
    env.reset(seed=4)
    churns = []
    for action in [1, 1, 1, 2, 0, 0]:
        _, _, done, info = env.step(action)
        churns.append(info["churn"])
        if done:
            break
    assert max(churns[3:]) >= max(churns[:3])


def test_onboarding_helps_young_more_than_old():
    cfg = HorizonRevConfig.base_case()
    env_young = HorizonRevEnv(cfg)
    env_old = HorizonRevEnv(cfg)
    env_young.reset(seed=9)
    env_old.reset(seed=9)

    # Force young-heavy vs old-heavy cohorts with same totals.
    env_young._state["cohorts"]["SMB"][:] = 0.0
    env_young._state["cohorts"]["ENT"][:] = 0.0
    env_young._state["cohorts"]["SMB"][0] = 250.0
    env_young._state["cohorts"]["SMB"][1] = 220.0
    env_young._state["cohorts"]["ENT"][0] = 170.0
    env_young._state["cohorts"]["ENT"][1] = 160.0

    env_old._state["cohorts"]["SMB"][:] = 0.0
    env_old._state["cohorts"]["ENT"][:] = 0.0
    env_old._state["cohorts"]["SMB"][-1] = 470.0
    env_old._state["cohorts"]["ENT"][-1] = 330.0

    _, _, _, info_young_onb = env_young.step(4)
    _, _, _, info_old_onb = env_old.step(4)
    assert info_young_onb["churn"] <= info_old_onb["churn"]


def test_monte_carlo_runner_outputs_valid_metrics():
    result = run_monte_carlo(
        env_factory=lambda: HorizonRevEnv(HorizonRevConfig.base_case()),
        policy_fn=lambda obs, env: heuristic_action(obs),
        seeds=range(1, 11),
        n_episodes=12,
    )
    assert result["total_reward_distribution"]
    assert len(result["total_reward_distribution"]) == 12
    assert result["p10_total_reward"] <= result["mean_total_reward"] or np.isclose(
        result["p10_total_reward"], result["mean_total_reward"]
    )
    assert 0.0 <= result["churn_spike_probability"] <= 1.0
