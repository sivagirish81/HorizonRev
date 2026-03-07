"""Run one heuristic-controlled episode in HorizonRev."""

from __future__ import annotations

from horizonrev import HorizonRevConfig, HorizonRevEnv
from horizonrev.dynamics.experiments import ACTION_NAMES, heuristic_action


def main() -> None:
    env = HorizonRevEnv(config=HorizonRevConfig.default())
    obs = env.reset(seed=42)
    done = False
    total_reward = 0.0
    month = 0

    while not done:
        action = heuristic_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        month += 1
        print(
            f"month={month} action={ACTION_NAMES[action]} reward={reward:.3f} "
            f"arr={info['arr']:.2f} conversion={info['conversion']:.3f} churn={info['churn']:.3f}"
        )
    print(f"Heuristic episode reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
