"""Run one random episode in HorizonRev."""

from __future__ import annotations

from horizonrev import HorizonRevConfig, HorizonRevEnv
from horizonrev.dynamics.experiments import ACTION_NAMES


def main() -> None:
    env = HorizonRevEnv(config=HorizonRevConfig.default())
    obs = env.reset(seed=7)
    done = False
    total_reward = 0.0
    step_idx = 0

    print("Initial observation:", obs)
    while not done:
        action = env.action_space.sample(env._rng) if hasattr(env.action_space, "sample") else 0
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
        step_idx += 1
        print(
            f"step={step_idx} action={ACTION_NAMES[int(action)]} reward={reward:.3f} "
            f"arr={info['arr']:.2f} churn={info['churn']:.3f} drift={info['drift_event']} "
            f"new={info['new_customers']:.1f} churned={info['churned_customers']:.1f} shocks={info['active_shocks']}"
        )
        if info["delayed_effects_applied"]:
            print(" delayed:", info["delayed_effects_applied"])

    print(f"Episode complete. Total reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
