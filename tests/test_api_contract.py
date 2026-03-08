from __future__ import annotations

import numpy as np

from horizonrev import HorizonRevConfig, HorizonRevEnv


def test_env_api_contract():
    env = HorizonRevEnv(config=HorizonRevConfig.default())
    obs = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (12,)
    assert obs.dtype == np.float32

    done = False
    steps = 0
    while not done:
        obs, reward, done, info = env.step(0)
        steps += 1
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        for key in [
            "month",
            "arr",
            "conversion",
            "churn",
            "action_name",
            "drift_event",
            "delayed_effects_applied",
            "reward_components",
            "segment_metrics",
            "total_customers",
            "churned_customers",
            "new_customers",
            "pct_young_customers",
            "avg_quality",
            "active_shocks",
            "cohort_summary",
        ]:
            assert key in info
    assert steps == 6
