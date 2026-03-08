# HorizonRev

HorizonRev is a lightweight, reusable reinforcement learning environment for long-horizon revenue strategy design. An agent makes monthly go-to-market decisions across six months while handling delayed churn effects and a mid-episode market drift event.

Project capabilities:

- Core environment package with minimal dependencies
- OpenEnv-compatible API (`reset`, `step`, `action_space`, `observation_space`)
- Hugging Face Spaces demo app (Gradio)
- Colab-friendly TRL PPO notebook
- Deterministic tests and examples
- Optional capped/uncapped token-scaled reward mode with quality gating

## Why HorizonRev

Most short-horizon simulators optimize immediate conversion. HorizonRev introduces delayed consequences:

- increasing discounts improves immediate conversion but schedules future churn penalties
- investing in onboarding reduces future churn after a delay
- at month 3, market drift changes SMB and ENT sensitivities

This creates a compact but meaningful long-horizon optimization task in only 6 steps.

## Installation

```bash
pip install -e .
```

Core dependencies are pinned in `requirements.txt`:

- `openenv-core==0.2.1` (OpenEnv stable 0.2.1 runtime distribution)
- `numpy`

## Reusability

Import `HorizonRevEnv` and select a config preset:

```python
from horizonrev import HorizonRevEnv, HorizonRevConfig

env = HorizonRevEnv(config=HorizonRevConfig.default())
obs = env.reset(seed=42)
```

Presets:

- `HorizonRevConfig.default()`
- `HorizonRevConfig.hard()`
- `HorizonRevConfig.no_drift()`

Reward mode options:

- `reward_mode="capped"`: token bonus is tightly bounded for stable scoring.
- `reward_mode="uncapped"`: token bonus accumulates across episode (still soft-saturated per step).

## Environment Spec

- **Episode length:** 6 steps (months 1..6)
- **Action space:** discrete(8)
  - `0 NOOP`
  - `1 INCREASE_DISCOUNT`
  - `2 DECREASE_DISCOUNT`
  - `3 LAUNCH_PRICING_AB`
  - `4 INVEST_ONBOARDING`
  - `5 SHIFT_SALES_SMB`
  - `6 SHIFT_SALES_ENT`
  - `7 STOP_PRICING_AB`
- **Observation vector (float32, normalized):**
  - `[month_norm, arr_norm, conv_norm, churn_norm, discount_norm, smb_demand_norm, ent_demand_norm, pricing_test_active, onboarding_invest_active, sales_focus, pct_young, avg_quality]`
- **Reward:**
  - `base_reward = (arr - prev_arr)/scale - alpha * churn - beta * confounding_penalty`
  - `reward = base_reward + lambda_token * token_bonus`
  - terminal bonus if ARR and churn meet thresholds

### Capped vs Uncapped Reward (Mercor sub-theme)

HorizonRev supports planning-aware reward scaling from agent reports while preventing verbosity exploits.

- `step(action, agent_report=None)` accepts optional per-step report text.
- `submit_report(text)` can be used before `step()` if your runner needs strict action-only step calls.
- Token count is lightweight (`len(text.split())`) and truncated at `report_word_cap` (default: 2000).
- Token bonus is quality-gated:
  - Required section markers:
    - `Hypothesis:`
    - `Action:`
    - `Expected Impact:`
    - `Risks:`
    - `Next Step:`
  - Must mention at least 2 environment metrics (`ARR`, `churn`, `conversion`, `discount`, `drift`, `month`).
  - If `planning_quality_score < quality_gate_threshold` (default `0.4`), token bonus is forced to `0`.
- Anti-spam controls:
  - soft token scaling (`log(1 + token_count)`) instead of linear growth
  - optional low-quality verbosity penalty for very long low-quality reports

Reward component fields in `info["reward_components"]`:

- `delta_arr_component`
- `churn_penalty`
- `confounding_penalty`
- `planning_quality_score`
- `token_count`
- `token_bonus`
- `token_bonus_component`
- `low_quality_verbosity_penalty`
- `base_reward`

`step()` returns an interpretable `info` payload with:

- `month`, `arr`, `conversion`, `churn`, `action_name`
- `drift_event`
- `delayed_effects_applied`
- `reward_components`
- `segment_metrics`

## Churn Model

HorizonRev uses cohort-based hazard modeling per segment (SMB and ENT), not a single global churn scalar.

- Internal state tracks active customers by cohort age bucket (`0..Amax`) and cohort quality mean.
- Baseline age hazard is configurable and calibratable:
  - `hazard_base(seg, age) = sigmoid(theta0_seg + theta_age_seg * log(1 + age))`
- Final hazard includes modifiers from onboarding, experiments, drift, shocks, promo-expiry effects, and cohort quality.
- Quality mechanism models discount-trap behavior:
  - high discount can lower quality of newly acquired cohorts
  - lower quality raises churn hazard over time
- Promo expiry effect:
  - a discount drop after high discount can trigger temporary churn spikes for promo-sensitive cohorts.

`step()` info includes:

- `total_customers`, `new_customers`, `churned_customers`
- `pct_young_customers`, `avg_quality`
- `active_shocks`
- `cohort_summary` (`age_0_2`, `age_3_5`, `age_6_plus`)

## Scenario Templates

Built-in presets in `HorizonRevConfig`:

- `base_case()`
- `optimistic()`
- `pessimistic()`
- `competitor_enters_early()`
- `macro_downturn()`

These presets adjust conversion, hazard parameters, drift behavior, traffic, and shock model probabilities/magnitudes.

## Monte Carlo Scenarios

Run strategy robustness checks across seeds:

```bash
python examples/monte_carlo_scenarios.py
```

The Monte Carlo runner (`horizonrev.monte_carlo.run_monte_carlo`) returns:

- mean/median/p10 total reward
- mean final ARR/churn
- churn spike probability
- raw episode distributions for downstream plotting

## Calibration to Real Data

HorizonRev is structured for later calibration from real retention and experiment logs.

Recommended calibration inputs:

1. Cohort retention table by account age (monthly)
2. Segment-level conversion and demand history
3. Experiment/intervention logs (discount, onboarding, pricing tests)
4. Known shock windows (competitor changes, outages, macro events)

Typical fitting workflow:

- fit `theta0_*` and `theta_age_*` to retention hazard shape
- fit quality and promo parameters using discount-policy cohorts
- fit shock parameters to event windows
- fit conversion sensitivities from experiment outcomes

## Run Examples

```bash
python examples/random_rollout.py
python examples/heuristic_rollout.py
python examples/monte_carlo_scenarios.py
```

## Run Tests

Install test dependency:

```bash
pip install -e ".[dev]"
```

Then run:

```bash
pytest
```

## Run HF Space App Locally

Install demo dependencies:

```bash
pip install -r requirements-space.txt
```

Run the app:

```bash
python apps/hf_space_app.py
```

For statistically robust demo comparison, use `Compare Random vs Heuristic/Trained`:

- compare now evaluates on unseen seeds (`1000..1099`)
- reports mean, std, and 95% CI (not just a single average)
- runs across multiple scenarios (`base_case`, `pessimistic`, `macro_downturn`)
- uses consistent report style for all agents during comparison
- charts include both step reward and cumulative reward

## Colab / TRL PPO Training

Open `notebooks/HorizonRev_TRL_Train.ipynb` in Colab.

Notebook includes:

1. package install (`requirements.txt` + `trl`, `transformers`, `torch`)
2. environment import and wrappers
3. minimal TRL PPO setup
4. short training loop
5. reward curve plotting
6. random vs trained reward comparison over 20 episodes

It also shows how to save policy artifacts for optional loading in the app.

For a direct Colab-friendly version that includes minimal-vs-structured report baselines in uncapped mode, use:

- `notebooks/HorizonRev_TRL_Train_Colab.ipynb`

## Recommended RL Evaluation Protocol

For reliable evidence beyond single-run outcomes:

- Train for `>= 1200` episodes
- Evaluate on unseen seeds (`>= 300`)
- Compare against both `Random` and `Heuristic`
- Report `mean`, `std`, and `95% CI`
- Test across scenarios (`base_case`, `pessimistic`, `macro_downturn`, `optimistic`)
- Keep reward mode and report style consistent between training and evaluation

Use the training and evaluation script:

```bash
pip install -r requirements-train.txt
pip install -e .
python scripts/train_mlp.py --train-episodes 1200 --eval-seed-count 300
```

Artifacts:

- policy weights: `artifacts/trained_policy_mlp.npz` (or custom `--policy-out`)
- policy metadata: `artifacts/policy_metadata.json` (policy type + policy path for app loading)
- metrics JSON: `artifacts/evaluation_metrics.json`

## Northflank Training Job

Use `Dockerfile.train` as the build image for a Northflank Job.

Recommended command:

```bash
python scripts/train_mlp.py \
  --train-episodes 2000 \
  --eval-seed-count 500 \
  --reward-mode uncapped \
  --report-style structured \
  --policy-out artifacts/trained_policy_mlp.npz \
  --policy-metadata-out artifacts/policy_metadata.json \
  --metrics-out artifacts/evaluation_metrics.json
```

After job completion, retrieve artifacts and place `trained_policy_mlp.npz` in repo root (or keep it under `artifacts/`) for app inference.

## Hugging Face Spaces Deployment (Gradio)

1. Create a new Space (SDK: Gradio)
2. Push repository contents
3. Set Space entrypoint to `apps/hf_space_app.py`
4. Use `requirements-space.txt` for dependencies
5. Verify charts, logs, and compare flow in UI

## Project Structure

```text
horizonrev/
  pyproject.toml
  README.md
  LICENSE
  requirements.txt
  requirements-space.txt
  requirements-train.txt
  Dockerfile.train
  src/
    horizonrev/
      __init__.py
      env.py
      config.py
      spaces.py
      reward.py
      monte_carlo.py
      rendering.py
      dynamics/
        __init__.py
        transition.py
        delayed.py
        drift.py
        experiments.py
      utils/
        __init__.py
        seeding.py
        normalize.py
  apps/
    hf_space_app.py
  examples/
    monte_carlo_scenarios.py
    random_rollout.py
    heuristic_rollout.py
  tests/
    test_api_contract.py
    test_cohort_and_monte_carlo.py
    test_determinism.py
    test_reward_nontrivial.py
  notebooks/
    HorizonRev_TRL_Train.ipynb
    HorizonRev_TRL_Train_Colab.ipynb
  scripts/
    train_ppo.py
    train_mlp.py
```
