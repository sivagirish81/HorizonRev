"""Train a NumPy MLP policy with REINFORCE and evaluate on unseen seeds.

This trainer is dependency-light and works on modern Python versions (e.g. 3.13)
without Rust/tokenizers builds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from horizonrev import HorizonRevConfig, HorizonRevEnv
from horizonrev.dynamics.experiments import ACTION_NAMES, heuristic_action


BASE_CONFIG = HorizonRevConfig.default()
PRIMITIVE_ACTIONS = len(ACTION_NAMES)
N_ACTIONS = PRIMITIVE_ACTIONS + PRIMITIVE_ACTIONS ** BASE_CONFIG.actions_per_month
OBS_DIM = 12
EPISODE_LENGTH = BASE_CONFIG.episode_length


def config_for_reward_mode(cfg: HorizonRevConfig, reward_mode: str) -> HorizonRevConfig:
    return HorizonRevConfig(**{**cfg.__dict__, "reward_mode": reward_mode})


def report_for_style(obs: np.ndarray, style: str) -> str:
    if style != "structured":
        return "Action selected."
    month = int(round(float(obs[0]) * EPISODE_LENGTH))
    return (
        f"Hypothesis: month {month} plan can improve ARR while controlling churn.\n"
        "Action: pick discount/onboarding/sales focus based on current metrics.\n"
        "Expected Impact: improve conversion and protect delayed churn and cohort quality.\n"
        "Risks: over-discounting can backfire under drift and raise delayed churn.\n"
        "Next Step: track ARR, churn, conversion, discount, drift, month, pct_young, and avg_quality."
    )


def stats(scores: List[float]) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": std,
        "ci95": float(ci95),
        "n": int(arr.size),
    }


class MlpPolicy:
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w1 = (rng.normal(0.0, 1.0, size=(obs_dim, hidden_dim)) / np.sqrt(obs_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = (rng.normal(0.0, 1.0, size=(hidden_dim, n_actions)) / np.sqrt(hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(n_actions, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / np.sum(e)

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(obs @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        probs = self._softmax(logits)
        return hidden, probs

    def sample_action(self, obs: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray, np.ndarray]:
        hidden, probs = self.forward(obs)
        action = int(rng.choice(N_ACTIONS, p=probs))
        return action, hidden, probs

    def greedy_action(self, obs: np.ndarray) -> int:
        _, probs = self.forward(obs)
        return int(np.argmax(probs))

    def update_episode(
        self,
        obs_list: List[np.ndarray],
        hidden_list: List[np.ndarray],
        probs_list: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        lr: float,
        gamma: float,
        entropy_coef: float,
        max_grad_norm: float,
    ) -> None:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running = rewards[i] + gamma * running
            returns[i] = running
        if returns.size > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        dw1 = np.zeros_like(self.w1)
        db1 = np.zeros_like(self.b1)
        dw2 = np.zeros_like(self.w2)
        db2 = np.zeros_like(self.b2)

        for obs, hidden, probs, action, adv in zip(obs_list, hidden_list, probs_list, actions, returns):
            one_hot = np.zeros(N_ACTIONS, dtype=np.float32)
            one_hot[action] = 1.0
            # Entropy regularization keeps exploration alive during longer runs.
            entropy_grad = -entropy_coef * (-np.log(np.clip(probs, 1e-8, 1.0)) - 1.0) * probs
            grad_logits = (one_hot - probs) * float(adv) + entropy_grad
            dw2 += np.outer(hidden, grad_logits)
            db2 += grad_logits
            dh = (1.0 - hidden**2) * (self.w2 @ grad_logits)
            dw1 += np.outer(obs, dh)
            db1 += dh

        if max_grad_norm > 0.0:
            grad_sq = float(
                np.sum(dw1 * dw1)
                + np.sum(db1 * db1)
                + np.sum(dw2 * dw2)
                + np.sum(db2 * db2)
            )
            grad_norm = float(np.sqrt(max(1e-12, grad_sq)))
            if grad_norm > max_grad_norm:
                clip_scale = max_grad_norm / grad_norm
                dw1 *= clip_scale
                db1 *= clip_scale
                dw2 *= clip_scale
                db2 *= clip_scale

        scale = 1.0 / max(1, len(actions))
        self.w1 += lr * dw1 * scale
        self.b1 += lr * db1 * scale
        self.w2 += lr * dw2 * scale
        self.b2 += lr * db2 * scale


def train(
    policy: MlpPolicy,
    reward_mode: str,
    report_style: str,
    episodes: int,
    train_seed_start: int,
    train_seed_count: int,
    learning_rate: float,
    min_learning_rate: float,
    gamma: float,
    entropy_coef_start: float,
    entropy_coef_end: float,
    max_grad_norm: float,
    log_every: int,
) -> List[float]:
    rewards: List[float] = []
    train_seeds = list(range(train_seed_start, train_seed_start + train_seed_count))
    rng = np.random.default_rng(123)

    for ep in range(episodes):
        progress = float(ep) / float(max(1, episodes - 1))
        entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress
        lr = learning_rate + (min_learning_rate - learning_rate) * progress

        seed = train_seeds[ep % len(train_seeds)]
        env = HorizonRevEnv(config_for_reward_mode(HorizonRevConfig.base_case(), reward_mode))
        obs = env.reset(seed=seed)
        done = False
        total = 0.0

        obs_list: List[np.ndarray] = []
        hidden_list: List[np.ndarray] = []
        probs_list: List[np.ndarray] = []
        actions: List[int] = []
        step_rewards: List[float] = []

        while not done:
            action, hidden, probs = policy.sample_action(obs, rng)
            report = report_for_style(obs, report_style)
            obs_next, reward, done, _ = env.step(action, agent_report=report)

            obs_list.append(obs.astype(np.float32))
            hidden_list.append(hidden.astype(np.float32))
            probs_list.append(probs.astype(np.float32))
            actions.append(action)
            step_rewards.append(float(reward))

            total += float(reward)
            obs = obs_next

        policy.update_episode(
            obs_list=obs_list,
            hidden_list=hidden_list,
            probs_list=probs_list,
            actions=actions,
            rewards=step_rewards,
            lr=lr,
            gamma=gamma,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )
        rewards.append(total)
        if (ep + 1) % log_every == 0:
            print(
                f"[train] ep={ep + 1}/{episodes} avg_reward_last_{log_every}={np.mean(rewards[-log_every:]):.3f} "
                f"lr={lr:.5f} entropy_coef={entropy_coef:.5f}"
            )
    return rewards


def evaluate_agent(
    agent: str,
    policy: MlpPolicy,
    reward_mode: str,
    report_style: str,
    seeds: List[int],
    cfg_factory: Callable[[], HorizonRevConfig],
) -> List[float]:
    scores: List[float] = []
    for seed in seeds:
        env = HorizonRevEnv(config_for_reward_mode(cfg_factory(), reward_mode))
        obs = env.reset(seed=seed)
        done = False
        total = 0.0
        step_rng = np.random.default_rng(seed)
        while not done:
            if agent == "random":
                action = int(step_rng.integers(0, N_ACTIONS))
            elif agent == "heuristic":
                action = heuristic_action(obs)
            else:
                action = policy.greedy_action(obs)
            env.submit_report(report_for_style(obs, report_style))
            obs, reward, done, _ = env.step(action)
            total += float(reward)
        scores.append(total)
    return scores


def run_evaluation(policy: MlpPolicy, reward_mode: str, report_style: str, eval_seeds: List[int]) -> Dict[str, dict]:
    scenarios: Dict[str, Callable[[], HorizonRevConfig]] = {
        "base_case": HorizonRevConfig.base_case,
        "pessimistic": HorizonRevConfig.pessimistic,
        "macro_downturn": HorizonRevConfig.macro_downturn,
        "optimistic": HorizonRevConfig.optimistic,
    }
    out: Dict[str, dict] = {}
    for scenario_name, cfg_factory in scenarios.items():
        random_scores = evaluate_agent("random", policy, reward_mode, report_style, eval_seeds, cfg_factory)
        heuristic_scores = evaluate_agent("heuristic", policy, reward_mode, report_style, eval_seeds, cfg_factory)
        trained_scores = evaluate_agent("trained", policy, reward_mode, report_style, eval_seeds, cfg_factory)
        out[scenario_name] = {
            "random": {**stats(random_scores), "scores": random_scores},
            "heuristic": {**stats(heuristic_scores), "scores": heuristic_scores},
            "trained": {**stats(trained_scores), "scores": trained_scores},
        }
        out[scenario_name]["delta_trained_vs_random"] = out[scenario_name]["trained"]["mean"] - out[scenario_name]["random"]["mean"]
        out[scenario_name]["delta_trained_vs_heuristic"] = (
            out[scenario_name]["trained"]["mean"] - out[scenario_name]["heuristic"]["mean"]
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HorizonRev NumPy MLP policy and run robust evaluation.")
    parser.add_argument("--reward-mode", choices=["capped", "uncapped"], default="uncapped")
    parser.add_argument("--report-style", choices=["minimal", "structured"], default="structured")
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--train-seed-count", type=int, default=800)
    parser.add_argument("--eval-seed-start", type=int, default=10_000)
    parser.add_argument("--eval-seed-count", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--min-learning-rate", type=float, default=0.0005)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--entropy-coef-start", type=float, default=0.02)
    parser.add_argument("--entropy-coef-end", type=float, default=0.003)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--policy-out", default="artifacts/trained_policy_mlp.npz")
    parser.add_argument("--metrics-out", default="artifacts/evaluation_metrics.json")
    parser.add_argument("--policy-metadata-out", default="artifacts/policy_metadata.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"train_episodes={args.train_episodes}, reward_mode={args.reward_mode}, report_style={args.report_style}")
    print(
        "schedule="
        f"lr({args.learning_rate}->{args.min_learning_rate}), "
        f"entropy({args.entropy_coef_start}->{args.entropy_coef_end}), "
        f"hidden_dim={args.hidden_dim}, max_grad_norm={args.max_grad_norm}"
    )
    policy = MlpPolicy(obs_dim=OBS_DIM, hidden_dim=args.hidden_dim, n_actions=N_ACTIONS, seed=args.seed)
    training_rewards = train(
        policy=policy,
        reward_mode=args.reward_mode,
        report_style=args.report_style,
        episodes=args.train_episodes,
        train_seed_start=args.train_seed_start,
        train_seed_count=args.train_seed_count,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        gamma=args.gamma,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
    )

    policy_out = Path(args.policy_out)
    policy_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(policy_out, w1=policy.w1, b1=policy.b1, w2=policy.w2, b2=policy.b2)
    print(f"saved_policy={policy_out}")

    policy_metadata = {
        "model_name": "numpy-mlp-reinforce",
        "policy_type": "mlp_numpy",
        "policy_path": str(policy_out),
        "obs_dim": OBS_DIM,
        "episode_length": EPISODE_LENGTH,
        "actions_per_month": BASE_CONFIG.actions_per_month,
        "hidden_dim": args.hidden_dim,
        "n_actions": N_ACTIONS,
        "reward_mode": args.reward_mode,
        "report_style": args.report_style,
    }
    policy_metadata_out = Path(args.policy_metadata_out)
    policy_metadata_out.parent.mkdir(parents=True, exist_ok=True)
    policy_metadata_out.write_text(json.dumps(policy_metadata, indent=2))
    print(f"saved_policy_metadata={policy_metadata_out}")

    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seed_count))
    metrics = {
        "config": vars(args),
        "training_reward_stats": stats(training_rewards),
        "evaluation": run_evaluation(
            policy=policy,
            reward_mode=args.reward_mode,
            report_style=args.report_style,
            eval_seeds=eval_seeds,
        ),
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))
    print(f"saved_metrics={metrics_out}")
    print(json.dumps({"training_reward_stats": metrics["training_reward_stats"]}, indent=2))


if __name__ == "__main__":
    main()
