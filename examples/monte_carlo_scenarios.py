"""Run Monte Carlo comparisons across standard HorizonRev scenarios."""

from __future__ import annotations

from horizonrev import HorizonRevConfig, HorizonRevEnv, run_monte_carlo
from horizonrev.dynamics.experiments import heuristic_action


def heuristic_policy(obs, _env):
    return heuristic_action(obs)


def _scenario_configs():
    return {
        "optimistic": HorizonRevConfig.optimistic(),
        "base_case": HorizonRevConfig.base_case(),
        "pessimistic": HorizonRevConfig.pessimistic(),
        "competitor_early": HorizonRevConfig.competitor_enters_early(),
        "macro_downturn": HorizonRevConfig.macro_downturn(),
    }


def main() -> None:
    seeds = list(range(10, 40))
    episodes = 40
    print(
        f"{'scenario':<20} {'mean_reward':>12} {'p10_reward':>12} "
        f"{'mean_final_arr':>14} {'mean_final_churn':>17} {'spike_prob':>11}"
    )
    print("-" * 92)
    for name, cfg in _scenario_configs().items():
        result = run_monte_carlo(
            env_factory=lambda cfg=cfg: HorizonRevEnv(cfg),
            policy_fn=heuristic_policy,
            seeds=seeds,
            n_episodes=episodes,
        )
        print(
            f"{name:<20} {result['mean_total_reward']:>12.3f} {result['p10_total_reward']:>12.3f} "
            f"{result['mean_final_arr']:>14.2f} {result['mean_final_churn']:>17.4f} "
            f"{result['churn_spike_probability']:>11.3f}"
        )


if __name__ == "__main__":
    main()
