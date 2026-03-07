"""Gradio app for HorizonRev, suitable for Hugging Face Spaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from horizonrev import HorizonRevConfig, HorizonRevEnv
from horizonrev.dynamics.experiments import ACTION_NAMES, heuristic_action
from horizonrev.rendering import format_step_log


@dataclass
class AppState:
    env: HorizonRevEnv = field(default_factory=lambda: HorizonRevEnv(HorizonRevConfig.default()))
    obs: np.ndarray | None = None
    done: bool = False
    logs: List[str] = field(default_factory=list)
    history: Dict[str, List[float]] = field(default_factory=lambda: {"month": [], "arr": [], "churn": [], "reward": []})


def _maybe_load_trained_weights():
    weights_path = Path("trained_policy.npy")
    if not weights_path.exists():
        return None
    arr = np.load(weights_path)
    if arr.shape != (10, 8):
        return None
    return arr


TRAINED_WEIGHTS = _maybe_load_trained_weights()


def _pick_action(agent_type: str, obs: np.ndarray, rng: np.random.Generator) -> int:
    if agent_type == "Random":
        return int(rng.integers(0, 8))
    if agent_type == "Heuristic":
        return heuristic_action(obs)
    if agent_type == "Trained" and TRAINED_WEIGHTS is not None:
        logits = obs @ TRAINED_WEIGHTS
        return int(np.argmax(logits))
    return heuristic_action(obs)


def _make_figure(history: Dict[str, List[float]]):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    months = history["month"] or [0]
    axes[0].plot(months, history["arr"] or [0.0], marker="o")
    axes[0].set_title("ARR")
    axes[0].set_xlabel("Month")
    axes[1].plot(months, history["churn"] or [0.0], marker="o", color="tab:orange")
    axes[1].set_title("Churn")
    axes[1].set_xlabel("Month")
    axes[2].plot(months, history["reward"] or [0.0], marker="o", color="tab:green")
    axes[2].set_title("Reward")
    axes[2].set_xlabel("Month")
    fig.tight_layout()
    return fig


def _reset(state: AppState):
    state.env = HorizonRevEnv(HorizonRevConfig.default())
    state.obs = state.env.reset(seed=123)
    state.done = False
    state.logs = ["Environment reset."]
    state.history = {"month": [], "arr": [], "churn": [], "reward": []}
    return state, _make_figure(state.history), "\n".join(state.logs)


def _step(agent_type: str, state: AppState):
    if state.obs is None:
        state, fig, log_text = _reset(state)
        return state, fig, log_text
    if state.done:
        state.logs.append("Episode already finished. Press Reset.")
        return state, _make_figure(state.history), "\n".join(state.logs)

    rng = np.random.default_rng(1234 + len(state.history["month"]))
    action = _pick_action(agent_type, state.obs, rng)
    obs, reward, done, info = state.env.step(action)
    state.obs = obs
    state.done = done
    state.history["month"].append(info["month"])
    state.history["arr"].append(info["arr"])
    state.history["churn"].append(info["churn"])
    state.history["reward"].append(reward)

    state.logs.append(format_step_log(info))
    if info["drift_event"]:
        state.logs.append("Market drift event triggered at month 3.")
    if info["delayed_effects_applied"]:
        state.logs.append(f"Delayed effects: {info['delayed_effects_applied']}")

    return state, _make_figure(state.history), "\n".join(state.logs)


def _run_episode(agent_type: str, state: AppState):
    if state.obs is None or state.done:
        state, _, _ = _reset(state)
    while not state.done:
        state, _, _ = _step(agent_type, state)
    return state, _make_figure(state.history), "\n".join(state.logs)


def _evaluate_agent(agent_type: str, n_episodes: int = 20) -> float:
    rewards = []
    for ep in range(n_episodes):
        env = HorizonRevEnv(HorizonRevConfig.default())
        obs = env.reset(seed=ep)
        done = False
        total = 0.0
        rng = np.random.default_rng(ep)
        while not done:
            action = _pick_action(agent_type, obs, rng)
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
    return float(np.mean(rewards))


def _compare(agent_type: str, state: AppState):
    baseline = _evaluate_agent("Random", n_episodes=20)
    contender = _evaluate_agent(agent_type if agent_type != "Random" else "Heuristic", n_episodes=20)
    lines = [
        f"Random avg reward (20 eps): {baseline:.3f}",
        f"{agent_type if agent_type != 'Random' else 'Heuristic'} avg reward (20 eps): {contender:.3f}",
        f"Improvement: {contender - baseline:+.3f}",
    ]
    state.logs.extend(lines)
    return state, _make_figure(state.history), "\n".join(state.logs)


with gr.Blocks(title="HorizonRev") as demo:
    gr.Markdown("# HorizonRev: Long-Horizon Revenue Strategy")
    gr.Markdown("Manage discounts, onboarding, and sales focus with delayed churn effects and market drift.")

    agent = gr.Dropdown(choices=["Random", "Heuristic", "Trained"], value="Heuristic", label="Agent Type")
    state = gr.State(AppState())
    plot = gr.Plot(label="Episode Metrics")
    logs = gr.Textbox(label="Logs", lines=16)

    with gr.Row():
        btn_reset = gr.Button("Reset")
        btn_step = gr.Button("Step")
        btn_run = gr.Button("Run Episode")
        btn_compare = gr.Button("Compare Random vs Heuristic/Trained")

    btn_reset.click(fn=_reset, inputs=[state], outputs=[state, plot, logs])
    btn_step.click(fn=_step, inputs=[agent, state], outputs=[state, plot, logs])
    btn_run.click(fn=_run_episode, inputs=[agent, state], outputs=[state, plot, logs])
    btn_compare.click(fn=_compare, inputs=[agent, state], outputs=[state, plot, logs])

    demo.load(fn=_reset, inputs=[state], outputs=[state, plot, logs])


if __name__ == "__main__":
    demo.launch()
