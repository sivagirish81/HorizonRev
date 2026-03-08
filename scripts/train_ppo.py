"""Train PPO and evaluate policy performance across unseen seeds/scenarios.

This script uses the classic TRL PPO stack (trl==0.11.4) for compatibility
with the notebook training workflow and the saved policy artifact format.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

from horizonrev import HorizonRevConfig, HorizonRevEnv
from horizonrev.dynamics.experiments import heuristic_action

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
except Exception as exc:  # pragma: no cover - import guard for friendly failure
    raise RuntimeError(
        "TRL PPO imports failed. Install dependencies from requirements-train.txt "
        "and run with Python 3.11/3.12."
    ) from exc


ACTIONS = list(range(8))
ACTION_TOKENS = [f"<A{i}>" for i in ACTIONS]


def obs_to_text(obs: np.ndarray) -> str:
    return (
        f"month={obs[0]:.3f} arr={obs[1]:.3f} conv={obs[2]:.3f} churn={obs[3]:.3f} "
        f"discount={obs[4]:.3f} smb_dem={obs[5]:.3f} ent_dem={obs[6]:.3f} "
        f"pricing={int(obs[7])} onboarding={int(obs[8])} focus={int(obs[9])} "
        f"pct_young={obs[10]:.3f} avg_quality={obs[11]:.3f}. "
        f"Emit one action token from: {' '.join(ACTION_TOKENS)}"
    )


def decode_action(text: str, rng: random.Random) -> int:
    for i, tok in enumerate(ACTION_TOKENS):
        if tok in text:
            return i
    return rng.randint(0, 7)


def minimal_report(_obs: np.ndarray) -> str:
    return "Action selected."


def structured_report(obs: np.ndarray) -> str:
    month = int(round(float(obs[0]) * 6))
    return (
        f"Hypothesis: month {month} plan can improve ARR while controlling churn.\n"
        "Action: pick discount/onboarding/sales focus based on current metrics.\n"
        "Expected Impact: improve conversion and protect delayed churn and cohort quality.\n"
        "Risks: over-discounting can backfire under drift and raise delayed churn.\n"
        "Next Step: track ARR, churn, conversion, discount, drift, month, pct_young, and avg_quality."
    )


def report_for_style(obs: np.ndarray, style: str) -> str:
    return structured_report(obs) if style == "structured" else minimal_report(obs)


def config_for_reward_mode(cfg: HorizonRevConfig, reward_mode: str) -> HorizonRevConfig:
    return HorizonRevConfig(**{**cfg.__dict__, "reward_mode": reward_mode})


def stats(scores: List[float]) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = (1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "std": std,
        "ci95": float(ci95),
        "n": int(len(arr)),
    }


def make_model_and_trainer(model_name: str, device: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Keep PPO batch_size aligned with HorizonRev episode length (6 by default).
    # TRL step() expects exactly batch_size query/response/reward entries.
    ppo_batch_size = HorizonRevConfig.base_case().episode_length
    ppo_trainer = PPOTrainer(
        config=PPOConfig(
            batch_size=ppo_batch_size,
            mini_batch_size=2,
            learning_rate=1e-5,
            ppo_epochs=2,
            log_with=None,
        ),
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
    )
    return model, tokenizer, ppo_trainer


def train(
    ppo_trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    reward_mode: str,
    report_style: str,
    episodes: int,
    train_seed_start: int,
    train_seed_count: int,
    log_every: int,
) -> List[float]:
    rewards: List[float] = []
    py_rng = random.Random(123)
    train_seeds = list(range(train_seed_start, train_seed_start + train_seed_count))
    for ep in range(episodes):
        seed = train_seeds[ep % len(train_seeds)]
        env = HorizonRevEnv(config_for_reward_mode(HorizonRevConfig.base_case(), reward_mode))
        obs = env.reset(seed=seed)
        done = False
        total = 0.0
        query_tensors = []
        response_tensors = []
        reward_tensors = []

        while not done:
            # TRL PPOTrainer.generate expects a 1D query tensor (seq_len), not batched input.
            q = tokenizer(obs_to_text(obs), return_tensors="pt").input_ids.squeeze(0).to(
                ppo_trainer.model.pretrained_model.device
            )
<<<<<<< HEAD
            q_attention_mask = torch.ones_like(q).unsqueeze(0)
=======
            q_attention_mask = torch.ones_like(q)
>>>>>>> f443d0d (Updated generation to be called with attention_mask and pad_token_id)
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            r = ppo_trainer.generate(
                q,
                attention_mask=q_attention_mask,
                pad_token_id=pad_token_id,
                max_new_tokens=4,
                do_sample=True,
                top_k=0,
                top_p=1.0,
            )
            response_tokens = r[0][q.shape[-1] :] if r.ndim == 2 else r[q.shape[-1] :]
            generated = tokenizer.decode(response_tokens, skip_special_tokens=False)
            action = decode_action(generated, py_rng)
            report = report_for_style(obs, report_style)
            obs, reward, done, _ = env.step(action, agent_report=report)
            total += reward

            query_tensors.append(q)
            response_tensors.append(response_tokens)
            reward_tensors.append(torch.tensor(reward, dtype=torch.float32).to(ppo_trainer.model.pretrained_model.device))

        batch_size = int(ppo_trainer.config.batch_size)
        if len(query_tensors) < batch_size:
            print(
                f"[train] skipping PPO step at ep={ep + 1}: "
                f"collected={len(query_tensors)} < batch_size={batch_size}"
            )
        else:
            for start in range(0, len(query_tensors), batch_size):
                end = start + batch_size
                if end > len(query_tensors):
                    break
                ppo_trainer.step(
                    query_tensors[start:end],
                    response_tensors[start:end],
                    reward_tensors[start:end],
                )
        rewards.append(float(total))
        if (ep + 1) % log_every == 0:
            last_k = rewards[-log_every:]
            print(f"[train] ep={ep + 1}/{episodes} avg_reward_last_{log_every}={np.mean(last_k):.3f}")
    return rewards


def evaluate_agent(
    agent: str,
    model,
    tokenizer,
    reward_mode: str,
    report_style: str,
    seeds: List[int],
    cfg_factory: Callable[[], HorizonRevConfig],
) -> List[float]:
    scores: List[float] = []
    py_rng = random.Random(999)
    for seed in seeds:
        env = HorizonRevEnv(config_for_reward_mode(cfg_factory(), reward_mode))
        obs = env.reset(seed=seed)
        done = False
        total = 0.0
        step_rng = np.random.default_rng(seed)
        while not done:
            if agent == "random":
                action = int(step_rng.integers(0, 8))
            elif agent == "heuristic":
                action = heuristic_action(obs)
            else:
                q = tokenizer(obs_to_text(obs), return_tensors="pt").input_ids.to(model.pretrained_model.device)
                q_attention_mask = torch.ones_like(q)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                with torch.no_grad():
                    r = model.generate(
                        q,
                        attention_mask=q_attention_mask,
                        pad_token_id=pad_token_id,
                        max_new_tokens=4,
                        do_sample=False,
                    )
                action_text = tokenizer.decode(r[0][q.shape[-1] :], skip_special_tokens=False)
                action = decode_action(action_text, py_rng)
            env.submit_report(report_for_style(obs, report_style))
            obs, reward, done, _ = env.step(action)
            total += reward
        scores.append(float(total))
    return scores


def run_evaluation(model, tokenizer, reward_mode: str, report_style: str, eval_seeds: List[int]) -> Dict[str, dict]:
    scenarios: Dict[str, Callable[[], HorizonRevConfig]] = {
        "base_case": HorizonRevConfig.base_case,
        "pessimistic": HorizonRevConfig.pessimistic,
        "macro_downturn": HorizonRevConfig.macro_downturn,
        "optimistic": HorizonRevConfig.optimistic,
    }
    out: Dict[str, dict] = {}
    for scenario_name, cfg_factory in scenarios.items():
        random_scores = evaluate_agent("random", model, tokenizer, reward_mode, report_style, eval_seeds, cfg_factory)
        heuristic_scores = evaluate_agent("heuristic", model, tokenizer, reward_mode, report_style, eval_seeds, cfg_factory)
        trained_scores = evaluate_agent("trained", model, tokenizer, reward_mode, report_style, eval_seeds, cfg_factory)
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
    parser = argparse.ArgumentParser(description="Train HorizonRev PPO and run evaluation across scenarios.")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--reward-mode", choices=["capped", "uncapped"], default="uncapped")
    parser.add_argument("--report-style", choices=["minimal", "structured"], default="structured")
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--train-seed-count", type=int, default=800)
    parser.add_argument("--eval-seed-start", type=int, default=10_000)
    parser.add_argument("--eval-seed-count", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--policy-out", default="horizonrev_trl_policy.pt")
    parser.add_argument("--policy-metadata-out", default="artifacts/policy_metadata.json")
    parser.add_argument("--metrics-out", default="artifacts/evaluation_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"train_episodes={args.train_episodes}, reward_mode={args.reward_mode}, report_style={args.report_style}")

    model, tokenizer, ppo_trainer = make_model_and_trainer(model_name=args.model_name, device=device)
    training_rewards = train(
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_mode=args.reward_mode,
        report_style=args.report_style,
        episodes=args.train_episodes,
        train_seed_start=args.train_seed_start,
        train_seed_count=args.train_seed_count,
        log_every=args.log_every,
    )

    policy_out = Path(args.policy_out)
    policy_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), policy_out)
    print(f"saved_policy={policy_out}")
    policy_metadata = {
        "model_name": args.model_name,
        "policy_path": str(policy_out),
        "action_tokens": ACTION_TOKENS,
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
            model=model,
            tokenizer=tokenizer,
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
