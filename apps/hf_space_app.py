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
    reward_mode: str = "capped"


ACTION_TOKENS = [f"<A{i}>" for i in range(8)]


def _obs_to_text(obs: np.ndarray) -> str:
    return (
        f"month={obs[0]:.3f} arr={obs[1]:.3f} conv={obs[2]:.3f} churn={obs[3]:.3f} "
        f"discount={obs[4]:.3f} smb_dem={obs[5]:.3f} ent_dem={obs[6]:.3f} "
        f"pricing={int(obs[7])} onboarding={int(obs[8])} focus={int(obs[9])}. "
        f"Emit one action token from: {' '.join(ACTION_TOKENS)}"
    )


def _decode_action(text: str) -> int:
    for idx, tok in enumerate(ACTION_TOKENS):
        if tok in text:
            return idx
    return -1


class _TrlPolicy:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def pick_action(self, obs: np.ndarray) -> int:
        prompt = _obs_to_text(obs)
        model_device = next(self.model.parameters()).device
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(model_device)
        with np.errstate(all="ignore"):
            with __import__("torch").no_grad():
                out = self.model.generate(input_ids, max_new_tokens=4, do_sample=False)
        generated = self.tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=False)
        return _decode_action(generated)


def _maybe_load_trained_weights():
    weights_path = Path("trained_policy.npy")
    if not weights_path.exists():
        return None
    arr = np.load(weights_path)
    if arr.shape != (12, 8):
        return None
    return arr


TRAINED_WEIGHTS = _maybe_load_trained_weights()
TRL_POLICY_LOAD_ERROR: str | None = None


def _maybe_load_trl_policy():
    global TRL_POLICY_LOAD_ERROR
    pt_path = Path("horizonrev_trl_policy.pt")
    if not pt_path.exists():
        TRL_POLICY_LOAD_ERROR = "horizonrev_trl_policy.pt not found"
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        TRL_POLICY_LOAD_ERROR = f"Dependency import failed: {exc!r}"
        return None

    try:
        model_name = "sshleifer/tiny-gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        state_dict = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        TRL_POLICY_LOAD_ERROR = None
        return _TrlPolicy(model, tokenizer)
    except Exception as exc:
        TRL_POLICY_LOAD_ERROR = f"Policy load failed: {exc!r}"
        return None


TRAINED_TRL_POLICY = _maybe_load_trl_policy()
HAS_TRL_PT_FILE = Path("horizonrev_trl_policy.pt").exists()


def _trained_backend_status() -> str:
    if TRAINED_WEIGHTS is not None:
        return "trained_policy.npy"
    if TRAINED_TRL_POLICY is not None:
        return "horizonrev_trl_policy.pt"
    if HAS_TRL_PT_FILE:
        detail = TRL_POLICY_LOAD_ERROR or "unknown error"
        return f"horizonrev_trl_policy.pt found but not loadable: {detail}"
    return "none (Trained falls back to heuristic)"


def _pick_action(agent_type: str, obs: np.ndarray, rng: np.random.Generator) -> int:
    if agent_type == "Random":
        return int(rng.integers(0, 8))
    if agent_type == "Heuristic":
        return heuristic_action(obs)
    if agent_type == "Trained":
        if TRAINED_WEIGHTS is not None:
            logits = obs @ TRAINED_WEIGHTS
            return int(np.argmax(logits))
        if TRAINED_TRL_POLICY is not None:
            action = TRAINED_TRL_POLICY.pick_action(obs)
            if 0 <= action <= 7:
                return int(action)
    return heuristic_action(obs)


def _heuristic_report(obs: np.ndarray) -> str:
    month = int(round(float(obs[0]) * 6))
    arr = float(obs[1])
    conv = float(obs[2])
    churn = float(obs[3])
    discount = float(obs[4])
    pct_young = float(obs[10])
    avg_quality = float(obs[11])
    return (
        f"Hypothesis: At month {month}, improving conversion without harming churn can raise ARR.\n"
        f"Action: Balance discount and onboarding while adjusting sales focus by month.\n"
        f"Expected Impact: ARR improves from stronger conversion with controlled churn after delay.\n"
        f"Risks: High discount can raise delayed churn; drift can shift response sensitivity.\n"
        f"Next Step: Track ARR, churn, conversion, discount, drift, and month for next action.\n"
        f"Current snapshot: arr={arr:.3f}, conversion={conv:.3f}, churn={churn:.3f}, discount={discount:.3f}, "
        f"pct_young={pct_young:.3f}, avg_quality={avg_quality:.3f}."
    )


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


def _component_text(info: dict | None) -> str:
    if not info:
        return "No step yet."
    c = info["reward_components"]
    return (
        f"reward_mode={info.get('reward_mode')}\n"
        f"planning_quality_score={c.get('planning_quality_score', 0.0):.3f}\n"
        f"token_count={c.get('token_count', 0)}\n"
        f"token_bonus={c.get('token_bonus', 0.0):.3f}\n"
        f"token_bonus_component={c.get('token_bonus_component', 0.0):.3f}\n"
        f"delta_arr_component={c.get('delta_arr_component', 0.0):.3f}\n"
        f"churn_penalty={c.get('churn_penalty', 0.0):.3f}\n"
        f"confounding_penalty={c.get('confounding_penalty', 0.0):.3f}\n"
        f"churn_volatility_penalty={c.get('churn_volatility_penalty', 0.0):.3f}\n"
        f"episode_reward_total={info.get('episode_reward_total', 0.0):.3f}"
    )


def _reset(state: AppState, reward_mode: str):
    config = HorizonRevConfig.default()
    config = HorizonRevConfig(**{**config.__dict__, "reward_mode": reward_mode})
    state.env = HorizonRevEnv(config)
    state.obs = state.env.reset(seed=123)
    state.done = False
    state.reward_mode = reward_mode
    state.logs = ["Environment reset.", f"Trained backend: {_trained_backend_status()}"]
    state.history = {"month": [], "arr": [], "churn": [], "reward": []}
    return state, _make_figure(state.history), "\n".join(state.logs), "No step yet."


def _step(agent_type: str, reward_mode: str, report_text: str, state: AppState):
    if state.obs is None:
        state, fig, log_text, comp_text = _reset(state, reward_mode)
        return state, fig, log_text, comp_text
    if state.done:
        state.logs.append("Episode already finished. Press Reset.")
        return state, _make_figure(state.history), "\n".join(state.logs), "Episode complete."
    if state.reward_mode != reward_mode:
        state, fig, log_text, comp_text = _reset(state, reward_mode)
        return state, fig, log_text, comp_text

    rng = np.random.default_rng(1234 + len(state.history["month"]))
    action = _pick_action(agent_type, state.obs, rng)
    final_report = report_text.strip()
    if not final_report and agent_type == "Heuristic":
        final_report = _heuristic_report(state.obs)
    if final_report:
        state.env.submit_report(final_report)
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
    if info["active_shocks"]:
        state.logs.append(f"Active shocks: {info['active_shocks']}")
    state.logs.append(
        f"Customers={info['total_customers']:.1f} new={info['new_customers']:.1f} "
        f"churned={info['churned_customers']:.1f} young={info['pct_young_customers']:.3f} "
        f"quality={info['avg_quality']:.3f} cohorts={info['cohort_summary']}"
    )
    state.logs.append(
        f"Planning quality={info['reward_components'].get('planning_quality_score', 0.0):.3f} | "
        f"Token bonus={info['reward_components'].get('token_bonus', 0.0):.3f}"
    )

    return state, _make_figure(state.history), "\n".join(state.logs), _component_text(info)


def _run_episode(agent_type: str, reward_mode: str, report_text: str, state: AppState):
    if state.obs is None or state.done:
        state, _, _, _ = _reset(state, reward_mode)
    while not state.done:
        state, _, _, _ = _step(agent_type, reward_mode, report_text, state)
    return state, _make_figure(state.history), "\n".join(state.logs), "Episode complete."


def _evaluate_agent(agent_type: str, reward_mode: str, report_style: str, n_episodes: int = 20) -> float:
    config = HorizonRevConfig.default()
    config = HorizonRevConfig(**{**config.__dict__, "reward_mode": reward_mode})
    rewards = []
    for ep in range(n_episodes):
        env = HorizonRevEnv(config)
        obs = env.reset(seed=ep)
        done = False
        total = 0.0
        rng = np.random.default_rng(ep)
        while not done:
            action = _pick_action(agent_type, obs, rng)
            if report_style == "structured":
                env.submit_report(_heuristic_report(obs))
            else:
                env.submit_report("Action selected.")
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
    return float(np.mean(rewards))


def _compare(agent_type: str, reward_mode: str, state: AppState):
    baseline = _evaluate_agent("Random", reward_mode=reward_mode, report_style="minimal", n_episodes=20)
    contender = _evaluate_agent(
        agent_type if agent_type != "Random" else "Heuristic",
        reward_mode=reward_mode,
        report_style="structured",
        n_episodes=20,
    )
    lines = [
        f"Reward mode: {reward_mode}",
        f"Random avg reward (20 eps): {baseline:.3f}",
        f"{agent_type if agent_type != 'Random' else 'Heuristic'} avg reward (20 eps, structured report): {contender:.3f}",
        f"Improvement: {contender - baseline:+.3f}",
    ]
    state.logs.extend(lines)
    return state, _make_figure(state.history), "\n".join(state.logs), "See logs for compare output."


with gr.Blocks(title="HorizonRev") as demo:
    gr.Markdown("# HorizonRev: Long-Horizon Revenue Strategy")
    gr.Markdown("Manage discounts, onboarding, and sales focus with delayed churn effects and market drift.")

    agent = gr.Dropdown(choices=["Random", "Heuristic", "Trained"], value="Heuristic", label="Agent Type")
    reward_mode = gr.Radio(choices=["capped", "uncapped"], value="capped", label="Reward Mode")
    report = gr.Textbox(
        label="Agent Report (optional, structured reports improve token bonus)",
        lines=8,
        placeholder="Hypothesis: ...\nAction: ...\nExpected Impact: ...\nRisks: ...\nNext Step: ...",
    )
    state = gr.State(AppState())
    plot = gr.Plot(label="Episode Metrics")
    logs = gr.Textbox(label="Logs", lines=16)
    reward_components = gr.Textbox(label="Latest Reward Components", lines=9)

    with gr.Row():
        btn_reset = gr.Button("Reset")
        btn_step = gr.Button("Step")
        btn_run = gr.Button("Run Episode")
        btn_compare = gr.Button("Compare Random vs Heuristic/Trained")

    btn_reset.click(fn=_reset, inputs=[state, reward_mode], outputs=[state, plot, logs, reward_components])
    btn_step.click(fn=_step, inputs=[agent, reward_mode, report, state], outputs=[state, plot, logs, reward_components])
    btn_run.click(fn=_run_episode, inputs=[agent, reward_mode, report, state], outputs=[state, plot, logs, reward_components])
    btn_compare.click(fn=_compare, inputs=[agent, reward_mode, state], outputs=[state, plot, logs, reward_components])

    demo.load(fn=_reset, inputs=[state, reward_mode], outputs=[state, plot, logs, reward_components])


if __name__ == "__main__":
    demo.launch()
