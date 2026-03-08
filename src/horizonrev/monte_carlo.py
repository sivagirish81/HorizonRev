"""Monte Carlo scenario runner utilities."""

from __future__ import annotations

from statistics import median
from typing import Callable, Iterable

import numpy as np

from horizonrev import HorizonRevEnv


def run_monte_carlo(
    env_factory: Callable[[], HorizonRevEnv],
    policy_fn: Callable[[np.ndarray, HorizonRevEnv], int],
    seeds: Iterable[int],
    n_episodes: int,
    churn_spike_threshold: float = 0.09,
) -> dict:
    total_rewards = []
    final_arrs = []
    final_churns = []
    churn_spike_flags = []

    seeds_list = list(seeds)
    for ep in range(n_episodes):
        seed = seeds_list[ep % len(seeds_list)]
        env = env_factory()
        obs = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        last_info = None
        spike = False
        while not done:
            action = int(policy_fn(obs, env))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            spike = spike or (info["churn"] > churn_spike_threshold)
            last_info = info
        total_rewards.append(float(total_reward))
        final_arrs.append(float(last_info["arr"]))
        final_churns.append(float(last_info["churn"]))
        churn_spike_flags.append(spike)

    arr = np.asarray(total_rewards, dtype=np.float64)
    p10 = float(np.percentile(arr, 10))
    return {
        "mean_total_reward": float(np.mean(arr)),
        "median_total_reward": float(median(total_rewards)),
        "p10_total_reward": p10,
        "mean_final_arr": float(np.mean(final_arrs)),
        "mean_final_churn": float(np.mean(final_churns)),
        "churn_spike_probability": float(np.mean(churn_spike_flags)),
        "total_reward_distribution": total_rewards,
        "final_arr_distribution": final_arrs,
        "final_churn_distribution": final_churns,
    }
