from __future__ import annotations

from horizonrev import HorizonRevConfig, HorizonRevEnv


def run_episode(actions: list[int]):
    config = HorizonRevConfig.default()
    env = HorizonRevEnv(config=config)
    env.reset(seed=5)
    rewards = []
    infos = []
    for action in actions:
        _, reward, done, info = env.step(action)
        rewards.append(reward)
        infos.append(info)
        if done:
            break
    return rewards, infos


def test_discount_has_delayed_cost():
    aggressive_discount_actions = [1, 1, 0, 0, 0, 0]
    conservative_actions = [4, 0, 4, 0, 0, 0]

    _, discount_infos = run_episode(aggressive_discount_actions)
    _, conservative_infos = run_episode(conservative_actions)

    discount_churn_month_3 = discount_infos[2]["churn"]
    conservative_churn_month_3 = conservative_infos[2]["churn"]
    assert discount_churn_month_3 >= conservative_churn_month_3

    applied_labels = [label for info in discount_infos for label in info["delayed_effects_applied"]]
    assert "discount_backfire" in applied_labels


def test_rewards_non_constant():
    rewards, _ = run_episode([1, 3, 4, 6, 2, 7])
    assert len(set(round(r, 6) for r in rewards)) > 1
