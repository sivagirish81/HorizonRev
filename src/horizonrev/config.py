"""Configuration presets for HorizonRev."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class HorizonRevConfig:
    episode_length: int = 6
    max_cohort_age: int = 24
    initial_arr: float = 120_000.0
    initial_customer_base: float = 800.0
    arr_scale_obs: float = 250_000.0
    reward_arr_scale: float = 2_500.0
    traffic_scale: float = 170.0
    arpa: float = 120.0

    base_conv_smb: float = 0.11
    base_conv_ent: float = 0.06

    theta0_smb: float = -2.2
    theta_age_smb: float = -0.55
    theta0_ent: float = -2.9
    theta_age_ent: float = -0.42
    hazard_max_base: float = 0.5
    hazard_max_final: float = 0.8
    onboarding_young_tau: float = 4.0
    onboarding_effect_smb: float = 0.028
    onboarding_effect_ent: float = 0.04
    experiment_churn_effect_smb: float = 0.003
    experiment_churn_effect_ent: float = 0.001
    quality_beta_smb: float = 0.35
    quality_beta_ent: float = 0.45
    quality_mu_smb: float = 0.18
    quality_mu_ent: float = 0.28
    quality_sigma_smb: float = 0.08
    quality_sigma_ent: float = 0.06
    discount_quality_sensitivity_smb: float = 0.5
    discount_quality_sensitivity_ent: float = 0.25
    onboarding_quality_boost: float = 0.1
    quality_min: float = -1.0
    quality_max: float = 1.0
    promo_discount_threshold: float = 0.35
    promo_tag_scale: float = 1.4
    promo_expiry_duration: int = 2
    promo_expiry_multiplier: float = 0.06
    promo_age_tau: float = 3.0

    discount_step: float = 0.1
    discount_effect_smb: float = 0.13
    discount_effect_ent: float = 0.07
    pricing_ab_lift: float = 0.015
    onboarding_churn_reduction: float = 0.01
    delayed_discount_churn_delta: float = 0.02
    delayed_onboarding_churn_delta: float = -0.018
    delay_months: int = 2

    noise_conv_std: float = 0.008
    noise_churn_std: float = 0.004
    traffic_smb: float = 135.0
    traffic_ent: float = 65.0
    arpa_smb: float = 95.0
    arpa_ent: float = 180.0

    drift_enabled: bool = True
    drift_month: Optional[int] = 3
    drift_randomize: bool = False
    drift_month_range: Tuple[int, int] = (2, 4)
    drift_strength: float = 1.0
    drift_discount_multiplier_smb: float = 1.35
    drift_onboarding_multiplier_ent: float = 1.45
    drift_demand_shift_smb: float = 0.12
    drift_demand_shift_ent: float = -0.08
    drift_shock_sensitivity_ent: float = 1.2

    shock_competitor_entry_prob: float = 0.08
    shock_budget_freeze_prob: float = 0.06
    shock_outage_prob: float = 0.05
    shock_macro_prob: float = 0.04
    shock_competitor_duration: int = 2
    shock_budget_duration: int = 2
    shock_outage_duration: int = 1
    shock_macro_duration: int = 3
    shock_competitor_demand_mult_smb: float = 0.9
    shock_competitor_demand_mult_ent: float = 0.92
    shock_competitor_churn_bump_smb: float = 0.015
    shock_competitor_churn_bump_ent: float = 0.008
    shock_budget_conv_mult_ent: float = 0.82
    shock_budget_churn_bump_ent: float = 0.01
    shock_outage_churn_bump_ent: float = 0.025
    shock_macro_demand_mult_smb: float = 0.9
    shock_macro_demand_mult_ent: float = 0.88
    shock_macro_churn_bump_all: float = 0.01

    reward_alpha: float = 1.6
    reward_beta: float = 0.4
    reward_mode: str = "capped"
    lambda_token: float = 0.1
    quality_gate_threshold: float = 0.4
    report_word_cap: int = 2000
    token_bonus_cap_capped: float = 0.5
    episode_reward_cap: float = 100.0
    low_quality_token_threshold: int = 700
    low_quality_verbosity_penalty: float = 0.08
    report_required_markers: Tuple[str, ...] = (
        "Hypothesis:",
        "Action:",
        "Expected Impact:",
        "Risks:",
        "Next Step:",
    )
    report_required_metrics: Tuple[str, ...] = (
        "arr",
        "churn",
        "conversion",
        "discount",
        "drift",
        "month",
    )
    long_pricing_test_months_threshold: int = 3
    long_pricing_test_penalty: float = 0.08
    confounding_penalty_if_dual: float = 0.12
    terminal_bonus: float = 1.75
    terminal_arr_threshold: float = 128_000.0
    terminal_churn_threshold: float = 0.05
    churn_volatility_penalty_weight: float = 0.25

    demand_smb_init: float = 1.0
    demand_ent_init: float = 1.0

    @classmethod
    def default(cls) -> "HorizonRevConfig":
        return cls.base_case()

    @classmethod
    def base_case(cls) -> "HorizonRevConfig":
        return cls()

    @classmethod
    def hard(cls) -> "HorizonRevConfig":
        return cls(
            noise_conv_std=0.015,
            noise_churn_std=0.008,
            drift_discount_multiplier_smb=1.6,
            drift_onboarding_multiplier_ent=1.7,
            drift_demand_shift_smb=0.2,
            drift_demand_shift_ent=-0.16,
            delayed_discount_churn_delta=0.024,
        )

    @classmethod
    def no_drift(cls) -> "HorizonRevConfig":
        return cls(drift_enabled=False)

    @classmethod
    def optimistic(cls) -> "HorizonRevConfig":
        return cls(
            base_conv_smb=0.13,
            base_conv_ent=0.08,
            shock_competitor_entry_prob=0.03,
            shock_budget_freeze_prob=0.03,
            shock_macro_prob=0.02,
            theta0_smb=-2.35,
            theta0_ent=-3.1,
            traffic_smb=150.0,
            traffic_ent=72.0,
        )

    @classmethod
    def pessimistic(cls) -> "HorizonRevConfig":
        return cls(
            base_conv_smb=0.09,
            base_conv_ent=0.045,
            theta0_smb=-1.95,
            theta0_ent=-2.6,
            shock_competitor_entry_prob=0.14,
            shock_budget_freeze_prob=0.12,
            shock_macro_prob=0.1,
            noise_conv_std=0.012,
            noise_churn_std=0.007,
            drift_strength=1.2,
        )

    @classmethod
    def competitor_enters_early(cls) -> "HorizonRevConfig":
        return cls(
            drift_month=2,
            drift_strength=1.1,
            shock_competitor_entry_prob=0.22,
            shock_competitor_duration=3,
            shock_competitor_demand_mult_smb=0.82,
            shock_competitor_demand_mult_ent=0.86,
        )

    @classmethod
    def macro_downturn(cls) -> "HorizonRevConfig":
        return cls(
            shock_macro_prob=0.2,
            shock_macro_duration=4,
            shock_macro_demand_mult_smb=0.78,
            shock_macro_demand_mult_ent=0.75,
            shock_macro_churn_bump_all=0.018,
            base_conv_smb=0.085,
            base_conv_ent=0.04,
        )
