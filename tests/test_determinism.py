from __future__ import annotations

import numpy as np

from horizonrev import HorizonRevConfig, HorizonRevEnv


def rollout(seed: int, actions: list[int]):
    env = HorizonRevEnv(config=HorizonRevConfig.default())
    obs = env.reset(seed=seed)
    out = [obs.copy()]
    rewards = []
    infos = []
    for action in actions:
        obs, reward, done, info = env.step(action)
        out.append(obs.copy())
        rewards.append(reward)
        infos.append(info)
        if done:
            break
    return out, rewards, infos


def test_seeded_rollout_is_deterministic():
    actions = [1, 3, 4, 6, 2, 7]
    obs_a, rew_a, info_a = rollout(11, actions)
    obs_b, rew_b, info_b = rollout(11, actions)

    for x, y in zip(obs_a, obs_b):
        assert np.allclose(x, y)
    assert np.allclose(rew_a, rew_b)
    assert [i["arr"] for i in info_a] == [i["arr"] for i in info_b]
    assert [i["churn"] for i in info_a] == [i["churn"] for i in info_b]
