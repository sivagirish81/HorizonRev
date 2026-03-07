"""Configuration presets for HorizonRev."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HorizonRevConfig:
    episode_length: int = 6
    initial_arr: float = 120_000.0
    initial_customer_base: float = 800.0
    arr_scale_obs: float = 250_000.0
    reward_arr_scale: float = 2_500.0
    traffic_scale: float = 170.0
    arpa: float = 120.0

    base_conv_smb: float = 0.11
    base_conv_ent: float = 0.06
    base_churn_smb: float = 0.045
    base_churn_ent: float = 0.03

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

    drift_enabled: bool = True
    drift_month: int = 3
    drift_discount_multiplier_smb: float = 1.35
    drift_onboarding_multiplier_ent: float = 1.45
    drift_demand_shift_smb: float = 0.12
    drift_demand_shift_ent: float = -0.08

    reward_alpha: float = 1.6
    reward_beta: float = 0.4
    long_pricing_test_months_threshold: int = 3
    long_pricing_test_penalty: float = 0.08
    confounding_penalty_if_dual: float = 0.12
    terminal_bonus: float = 1.75
    terminal_arr_threshold: float = 128_000.0
    terminal_churn_threshold: float = 0.05

    demand_smb_init: float = 1.0
    demand_ent_init: float = 1.0

    @classmethod
    def default(cls) -> "HorizonRevConfig":
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
