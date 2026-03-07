# HorizonRev

HorizonRev is a lightweight, reusable reinforcement learning environment for long-horizon revenue strategy design. An agent makes monthly go-to-market decisions across six months while handling delayed churn effects and a mid-episode market drift event.

Built for hackathons and rapid experimentation:

- Core environment package with minimal dependencies
- OpenEnv-compatible API (`reset`, `step`, `action_space`, `observation_space`)
- Hugging Face Spaces demo app (Gradio)
- Colab-friendly TRL PPO notebook
- Deterministic tests and examples

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

- `openenv==0.2.1`
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
  - `[month_norm, arr_norm, conv_norm, churn_norm, discount_norm, smb_demand_norm, ent_demand_norm, pricing_test_active, onboarding_invest_active, sales_focus]`
- **Reward:**
  - `(arr - prev_arr)/scale - alpha * churn - beta * confounding_penalty`
  - terminal bonus if ARR and churn meet thresholds

`step()` returns an interpretable `info` payload with:

- `month`, `arr`, `conversion`, `churn`, `action_name`
- `drift_event`
- `delayed_effects_applied`
- `reward_components`
- `segment_metrics`

## Run Examples

```bash
python examples/random_rollout.py
python examples/heuristic_rollout.py
```

## Run Tests

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
  src/
    horizonrev/
      __init__.py
      env.py
      config.py
      spaces.py
      reward.py
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
    random_rollout.py
    heuristic_rollout.py
  tests/
    test_api_contract.py
    test_determinism.py
    test_reward_nontrivial.py
  notebooks/
    HorizonRev_TRL_Train.ipynb
```
