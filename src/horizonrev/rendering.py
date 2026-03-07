"""Rendering and log formatting helpers."""

from __future__ import annotations


def format_step_log(info: dict) -> str:
    delayed = ", ".join(info.get("delayed_effects_applied", [])) or "none"
    drift_msg = "YES" if info.get("drift_event") else "no"
    return (
        f"Month {info.get('month')} | action={info.get('action_name')} | "
        f"ARR={info.get('arr'):.2f} | conv={info.get('conversion'):.3f} | "
        f"churn={info.get('churn'):.3f} | drift={drift_msg} | delayed={delayed}"
    )
