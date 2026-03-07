"""HorizonRev OpenEnv-compatible environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from horizonrev.config import HorizonRevConfig
from horizonrev.dynamics.delayed import make_queue, pop_effects
from horizonrev.dynamics.drift import apply_market_drift_if_needed
from horizonrev.dynamics.experiments import ACTION_NAMES
from horizonrev.dynamics.transition import (
    apply_action,
    compute_segment_metrics,
    create_initial_state,
    update_arr_and_base,
)
from horizonrev.reward import compute_confounding_penalty, compute_reward
from horizonrev.spaces import make_box, make_discrete
from horizonrev.utils.normalize import safe_norm
from horizonrev.utils.seeding import make_rng


class HorizonRevEnv:
    """A 6-step long-horizon revenue strategy environment."""

    metadata = {"name": "HorizonRevEnv", "render_modes": []}

    def __init__(self, config: HorizonRevConfig | None = None) -> None:
        self.config = config or HorizonRevConfig.default()
        self.action_space = make_discrete(8)
        self.observation_space = make_box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

        self._rng = make_rng(None)
        self._state = create_initial_state(self.config)
        self._delayed_queue = make_queue()

    def reset(self, seed: int | None = None):
        self._rng = make_rng(seed)
        self._state = create_initial_state(self.config)
        self._delayed_queue = make_queue()
        return self._get_observation()

    def step(self, action: int):
        self._validate_action(action)
        state = self._state

        drift_event = apply_market_drift_if_needed(state, self.config)
        delayed_delta, delayed_labels = pop_effects(self._delayed_queue, state["month"])

        apply_action(state, int(action), self.config, self._delayed_queue)
        segment_metrics = compute_segment_metrics(state, self.config, delayed_delta, self._rng)
        prev_arr, conversion, churn = update_arr_and_base(state, segment_metrics, self.config)

        is_terminal = state["month"] >= self.config.episode_length
        confounding_penalty = compute_confounding_penalty(
            pricing_test_active=state["pricing_test_active"],
            onboarding_invest_active=state["onboarding_invest_active"],
            pricing_test_months=state["pricing_test_months"],
            config=self.config,
        )
        reward, reward_components = compute_reward(
            prev_arr=prev_arr,
            new_arr=state["arr"],
            churn=churn,
            confounding_penalty=confounding_penalty,
            is_terminal=is_terminal,
            config=self.config,
        )

        info = {
            "month": int(state["month"]),
            "arr": float(state["arr"]),
            "conversion": float(conversion),
            "churn": float(churn),
            "action_name": ACTION_NAMES[int(action)],
            "drift_event": bool(drift_event),
            "delayed_effects_applied": delayed_labels,
            "reward_components": reward_components,
            "segment_metrics": segment_metrics,
        }

        if not is_terminal:
            state["month"] += 1

        obs = self._get_observation()
        done = bool(is_terminal)
        return obs, float(reward), done, info

    def _validate_action(self, action: int) -> None:
        if hasattr(self.action_space, "contains"):
            if not self.action_space.contains(int(action)):
                raise ValueError(f"Action {action} is out of bounds for action space")
        elif not 0 <= int(action) < 8:
            raise ValueError(f"Action {action} is out of bounds for action space")

    def _get_observation(self) -> np.ndarray:
        s = self._state
        obs = np.asarray(
            [
                safe_norm(float(s["month"]), float(self.config.episode_length)),
                safe_norm(float(s["arr"]), float(self.config.arr_scale_obs)),
                safe_norm(float(s["last_conversion"]), 0.45),
                safe_norm(float(s["last_churn"]), 0.2),
                safe_norm(float(s["discount_level"]), 1.0),
                safe_norm(float(s["smb_demand"]), 2.0),
                safe_norm(float(s["ent_demand"]), 2.0),
                1.0 if s["pricing_test_active"] else 0.0,
                1.0 if s["onboarding_invest_active"] else 0.0,
                float(s["sales_focus"]),
            ],
            dtype=np.float32,
        )
        return obs

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)
