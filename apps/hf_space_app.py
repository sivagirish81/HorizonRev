"""Gradio app for HorizonRev, suitable for Hugging Face Spaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from horizonrev import HorizonRevConfig, HorizonRevEnv
from horizonrev.dynamics.experiments import (
    ACTION_NAMES,
    DECREASE_DISCOUNT,
    INCREASE_DISCOUNT,
    LAUNCH_PRICING_AB,
    SHIFT_SALES_ENT,
    SHIFT_SALES_SMB,
    STOP_PRICING_AB,
    heuristic_action,
)
from horizonrev.rendering import format_step_log


@dataclass
class AppState:
    env: HorizonRevEnv = field(default_factory=lambda: HorizonRevEnv(HorizonRevConfig.default()))
    obs: np.ndarray | None = None
    done: bool = False
    logs: List[str] = field(default_factory=list)
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "month": [],
            "arr": [],
            "conversion": [],
            "churn": [],
            "reward": [],
            "cumulative_reward": [],
        }
    )
    reward_mode: str = "capped"


ACTION_TOKENS = [f"<A{i}>" for i in range(8)]
DEFAULT_TRL_MODEL_NAME = "sshleifer/tiny-gpt2"
DEFAULT_TRL_POLICY_PATH = Path("horizonrev_trl_policy.pt")
POLICY_METADATA_CANDIDATES = [Path("artifacts/policy_metadata.json"), Path("policy_metadata.json")]
MLP_POLICY_CANDIDATES = [Path("trained_policy_mlp.npz"), Path("artifacts/trained_policy_mlp.npz")]
DEFAULT_EPISODE_LENGTH = HorizonRevConfig.default().episode_length
PRIMITIVE_ACTION_COUNT = len(ACTION_NAMES)
ACTION_COUNT = PRIMITIVE_ACTION_COUNT + PRIMITIVE_ACTION_COUNT ** HorizonRevConfig.default().actions_per_month


def _decode_action_bundle(action: int, actions_per_month: int, primitive_actions: int) -> list[int]:
    if action < primitive_actions or actions_per_month <= 1:
        return [action]
    bundle_index = action - primitive_actions
    out: list[int] = []
    for power in range(actions_per_month - 1, -1, -1):
        base = primitive_actions**power
        token = bundle_index // base
        bundle_index %= base
        out.append(int(token))
    return out


def _is_conflicting_bundle(bundle: list[int]) -> bool:
    bundle_set = set(bundle)
    if INCREASE_DISCOUNT in bundle_set and DECREASE_DISCOUNT in bundle_set:
        return True
    if LAUNCH_PRICING_AB in bundle_set and STOP_PRICING_AB in bundle_set:
        return True
    if SHIFT_SALES_SMB in bundle_set and SHIFT_SALES_ENT in bundle_set:
        return True
    return False


@lru_cache(maxsize=8)
def _valid_action_indices(max_actions: int) -> tuple[int, ...]:
    valid: list[int] = []
    actions_per_month = HorizonRevConfig.default().actions_per_month
    for action in range(max_actions):
        bundle = _decode_action_bundle(action, actions_per_month, PRIMITIVE_ACTION_COUNT)
        if not _is_conflicting_bundle(bundle):
            valid.append(action)
    return tuple(valid)


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


class _NumpyMlpPolicy:
    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def pick_action(self, obs: np.ndarray) -> int:
        hidden = np.tanh(obs @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        max_actions = int(logits.shape[0])
        valid = _valid_action_indices(max_actions)
        if not valid:
            return int(np.argmax(logits))
        valid_arr = np.asarray(valid, dtype=np.int64)
        return int(valid_arr[int(np.argmax(logits[valid_arr]))])


def _load_policy_metadata() -> dict:
    for path in POLICY_METADATA_CANDIDATES:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data["_metadata_path"] = str(path)
                return data
        except Exception:
            continue
    return {}


def _resolve_policy_path(metadata: dict) -> Path:
    configured = metadata.get("policy_path")
    if isinstance(configured, str) and configured.strip():
        configured_path = Path(configured)
        if configured_path.exists():
            return configured_path
    return DEFAULT_TRL_POLICY_PATH


def _maybe_load_trained_weights():
    weights_path = Path("trained_policy.npy")
    if not weights_path.exists():
        return None
    arr = np.load(weights_path)
    if arr.shape != (12, 8):
        return None
    return arr


def _maybe_load_mlp_policy():
    for weights_path in MLP_POLICY_CANDIDATES:
        if not weights_path.exists():
            continue
        try:
            data = np.load(weights_path)
            w1 = np.asarray(data["w1"], dtype=np.float32)
            b1 = np.asarray(data["b1"], dtype=np.float32)
            w2 = np.asarray(data["w2"], dtype=np.float32)
            b2 = np.asarray(data["b2"], dtype=np.float32)
            if w1.ndim != 2 or b1.ndim != 1 or w2.ndim != 2 or b2.ndim != 1:
                continue
            if w1.shape[0] != 12 or w1.shape[1] != b1.shape[0] or w2.shape[0] != b1.shape[0]:
                continue
            if w2.shape[1] != b2.shape[0]:
                continue
            if b2.shape[0] < PRIMITIVE_ACTION_COUNT or b2.shape[0] > ACTION_COUNT:
                continue
            return _NumpyMlpPolicy(w1=w1, b1=b1, w2=w2, b2=b2), weights_path
        except Exception:
            continue
    return None, None


TRAINED_WEIGHTS = _maybe_load_trained_weights()
TRAINED_MLP_POLICY, TRAINED_MLP_POLICY_PATH = _maybe_load_mlp_policy()
TRL_POLICY_LOAD_ERROR: str | None = None
POLICY_METADATA = _load_policy_metadata()
TRL_POLICY_PATH = _resolve_policy_path(POLICY_METADATA)
TRL_MODEL_NAME = str(POLICY_METADATA.get("model_name", DEFAULT_TRL_MODEL_NAME))
POLICY_TYPE = str(POLICY_METADATA.get("policy_type", "trl_causal_lm"))


def _maybe_load_trl_policy():
    global TRL_POLICY_LOAD_ERROR
    if POLICY_TYPE != "trl_causal_lm":
        TRL_POLICY_LOAD_ERROR = f"policy_type={POLICY_TYPE} is not TRL"
        return None
    pt_path = TRL_POLICY_PATH
    if not pt_path.exists():
        TRL_POLICY_LOAD_ERROR = f"{pt_path} not found"
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        TRL_POLICY_LOAD_ERROR = f"Dependency import failed: {exc!r}"
        return None

    try:
        model_name = TRL_MODEL_NAME
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
HAS_TRL_PT_FILE = TRL_POLICY_PATH.exists()


def _trained_backend_status() -> str:
    if TRAINED_WEIGHTS is not None:
        return "trained_policy.npy"
    if TRAINED_MLP_POLICY is not None and TRAINED_MLP_POLICY_PATH is not None:
        return str(TRAINED_MLP_POLICY_PATH)
    if TRAINED_TRL_POLICY is not None:
        return f"{TRL_POLICY_PATH} (model={TRL_MODEL_NAME})"
    if HAS_TRL_PT_FILE:
        detail = TRL_POLICY_LOAD_ERROR or "unknown error"
        return f"{TRL_POLICY_PATH} found but not loadable (model={TRL_MODEL_NAME}): {detail}"
    return "none (Trained falls back to heuristic)"


def _pick_action(agent_type: str, obs: np.ndarray, rng: np.random.Generator) -> int:
    if agent_type == "Random":
        return int(rng.integers(0, ACTION_COUNT))
    if agent_type == "Heuristic":
        return heuristic_action(obs)
    if agent_type == "Trained":
        if TRAINED_WEIGHTS is not None:
            logits = obs @ TRAINED_WEIGHTS
            return int(np.argmax(logits))
        if TRAINED_MLP_POLICY is not None:
            return TRAINED_MLP_POLICY.pick_action(obs)
        if TRAINED_TRL_POLICY is not None:
            action = TRAINED_TRL_POLICY.pick_action(obs)
            if 0 <= action < ACTION_COUNT:
                return int(action)
    return heuristic_action(obs)


def _heuristic_report(obs: np.ndarray) -> str:
    month = int(round(float(obs[0]) * DEFAULT_EPISODE_LENGTH))
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    axes = axes.flatten()
    months = history["month"] or [0]
    axes[0].plot(months, history["arr"] or [0.0], marker="o")
    axes[0].set_title("ARR")
    axes[0].set_xlabel("Month")
    axes[1].plot(months, history["churn"] or [0.0], marker="o", color="tab:orange")
    axes[1].set_title("Churn")
    axes[1].set_xlabel("Month")
    axes[2].plot(months, history["reward"] or [0.0], marker="o", color="tab:green")
    axes[2].set_title("Step Reward")
    axes[2].set_xlabel("Month")
    axes[3].plot(months, history["cumulative_reward"] or [0.0], marker="o", color="tab:purple")
    axes[3].set_title("Cumulative Reward")
    axes[3].set_xlabel("Month")
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
    state.history = {"month": [], "arr": [], "conversion": [], "churn": [], "reward": [], "cumulative_reward": []}
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
    state.history["conversion"].append(info["conversion"])
    state.history["churn"].append(info["churn"])
    state.history["reward"].append(reward)
    running_total = reward if not state.history["cumulative_reward"] else state.history["cumulative_reward"][-1] + reward
    state.history["cumulative_reward"].append(float(running_total))

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
    final_total = state.history["cumulative_reward"][-1] if state.history["cumulative_reward"] else 0.0
    state.logs.append(f"Episode complete. Final cumulative reward={final_total:.3f}")
    return state, _make_figure(state.history), "\n".join(state.logs), "Episode complete."


def _report_for_style(obs: np.ndarray, report_style: str) -> str:
    if report_style == "structured":
        return _heuristic_report(obs)
    return "Action selected."


def _score_stats(scores: List[float]) -> dict:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "ci95": float(ci95), "n": int(arr.size)}


def _evaluate_agent_scores(
    agent_type: str,
    reward_mode: str,
    report_style: str,
    seeds: List[int],
    scenario: str,
) -> List[float]:
    scenario_map = {
        "base_case": HorizonRevConfig.base_case,
        "pessimistic": HorizonRevConfig.pessimistic,
        "macro_downturn": HorizonRevConfig.macro_downturn,
    }
    cfg_factory = scenario_map[scenario]
    rewards: List[float] = []
    for seed in seeds:
        config = cfg_factory()
        config = HorizonRevConfig(**{**config.__dict__, "reward_mode": reward_mode})
        env = HorizonRevEnv(config)
        obs = env.reset(seed=seed)
        done = False
        total = 0.0
        rng = np.random.default_rng(seed)
        while not done:
            action = _pick_action(agent_type, obs, rng)
            env.submit_report(_report_for_style(obs, report_style))
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(float(total))
    return rewards


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
            env.submit_report(_report_for_style(obs, report_style))
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
    return float(np.mean(rewards))


def _compare(agent_type: str, reward_mode: str, state: AppState):
    contender_name = agent_type if agent_type != "Random" else "Heuristic"
    eval_seeds = list(range(1000, 1100))
    scenarios = ["base_case", "pessimistic", "macro_downturn"]
    report_style = "structured"
    lines = [
        "=== Credible evaluation (unseen seeds) ===",
        f"Reward mode: {reward_mode}",
        f"Report style (all agents): {report_style}",
        f"Seeds: {eval_seeds[0]}..{eval_seeds[-1]} (n={len(eval_seeds)})",
    ]
    overall_random = []
    overall_contender = []
    for scenario in scenarios:
        random_scores = _evaluate_agent_scores(
            "Random", reward_mode=reward_mode, report_style=report_style, seeds=eval_seeds, scenario=scenario
        )
        contender_scores = _evaluate_agent_scores(
            contender_name, reward_mode=reward_mode, report_style=report_style, seeds=eval_seeds, scenario=scenario
        )
        overall_random.extend(random_scores)
        overall_contender.extend(contender_scores)
        random_stats = _score_stats(random_scores)
        contender_stats = _score_stats(contender_scores)
        delta_mean = contender_stats["mean"] - random_stats["mean"]
        lines.append(
            f"[{scenario}] Random mean={random_stats['mean']:.3f} +/-{random_stats['ci95']:.3f} "
            f"(std={random_stats['std']:.3f})"
        )
        lines.append(
            f"[{scenario}] {contender_name} mean={contender_stats['mean']:.3f} +/-{contender_stats['ci95']:.3f} "
            f"(std={contender_stats['std']:.3f})"
        )
        lines.append(f"[{scenario}] Delta mean ({contender_name}-Random) = {delta_mean:+.3f}")

    overall_random_stats = _score_stats(overall_random)
    overall_contender_stats = _score_stats(overall_contender)
    lines.append("---")
    lines.append(
        f"[overall] Random mean={overall_random_stats['mean']:.3f} +/-{overall_random_stats['ci95']:.3f} "
        f"(std={overall_random_stats['std']:.3f}, n={overall_random_stats['n']})"
    )
    lines.append(
        f"[overall] {contender_name} mean={overall_contender_stats['mean']:.3f} +/-{overall_contender_stats['ci95']:.3f} "
        f"(std={overall_contender_stats['std']:.3f}, n={overall_contender_stats['n']})"
    )
    lines.append(f"[overall] Delta mean ({contender_name}-Random) = {overall_contender_stats['mean'] - overall_random_stats['mean']:+.3f}")

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
