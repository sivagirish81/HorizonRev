"""HorizonRev OpenEnv-compatible environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from horizonrev.config import HorizonRevConfig
from horizonrev.dynamics.delayed import make_queue, pop_effects
from horizonrev.dynamics.drift import apply_market_drift_if_needed, initialize_drift_month
from horizonrev.dynamics.experiments import ACTION_NAMES
from horizonrev.dynamics.transition import (
    apply_action,
    compute_segment_metrics,
    create_initial_state,
    sample_and_update_shocks,
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
        if self.config.reward_mode not in {"capped", "uncapped"}:
            raise ValueError("reward_mode must be either 'capped' or 'uncapped'")
        self.action_space = make_discrete(8)
        self.observation_space = make_box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

        self._rng = make_rng(None)
        self._state = create_initial_state(self.config)
        self._delayed_queue = make_queue()
        self._pending_report = ""
        self._episode_reward_total = 0.0

    def reset(self, seed: int | None = None):
        self._rng = make_rng(seed)
        self._state = create_initial_state(self.config)
        initialize_drift_month(self._state, self.config, self._rng)
        self._delayed_queue = make_queue()
        self._pending_report = ""
        self._episode_reward_total = 0.0
        return self._get_observation()

    def submit_report(self, text: str) -> None:
        self._pending_report = text or ""

    def step(self, action: int, agent_report: str | None = None):
        self._validate_action(action)
        state = self._state
        report_text = self._pending_report if agent_report is None else (agent_report or "")
        self._pending_report = ""

        sample_and_update_shocks(state, self.config, self._rng)
        drift_event = apply_market_drift_if_needed(state, self.config)
        delayed_delta, delayed_labels = pop_effects(self._delayed_queue, state["month"])

        apply_action(state, int(action), self.config, self._delayed_queue)
        segment_metrics = compute_segment_metrics(state, self.config, delayed_delta, self._rng)
        prev_arr, conversion, churn, transition_details = update_arr_and_base(
            state=state,
            metrics=segment_metrics,
            delayed_churn_delta=delayed_delta,
            config=self.config,
            rng=self._rng,
        )

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
            churn_volatility=state["last_churn_volatility"],
            confounding_penalty=confounding_penalty,
            agent_report=report_text,
            is_terminal=is_terminal,
            config=self.config,
        )
        if self.config.reward_mode == "capped":
            remaining_budget = self.config.episode_reward_cap - self._episode_reward_total
            reward = min(reward, remaining_budget)
        self._episode_reward_total += float(reward)

        info = {
            "month": int(state["month"]),
            "arr": float(state["arr"]),
            "conversion": float(conversion),
            "churn": float(churn),
            "action_name": ACTION_NAMES[int(action)],
            "drift_event": bool(drift_event),
            "delayed_effects_applied": delayed_labels,
            "reward_components": reward_components,
            "segment_metrics": transition_details["segment_metrics"],
            "agent_report": report_text,
            "reward_mode": self.config.reward_mode,
            "episode_reward_total": float(self._episode_reward_total),
            "total_customers": float(state["total_customers"]),
            "churned_customers": float(state["churned_customers"]),
            "new_customers": float(state["new_customers"]),
            "pct_young_customers": float(state["pct_young_customers"]),
            "avg_quality": float(state["avg_quality"]),
            "active_shocks": sorted(list(state["active_shocks"].keys())),
            "cohort_summary": transition_details["cohort_summary"],
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
                safe_norm(float(s.get("pct_young_customers", 0.0)), 1.0),
                safe_norm(float(s.get("avg_quality", 0.0)) + 1.0, 2.0),
            ],
            dtype=np.float32,
        )
        return obs

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)
