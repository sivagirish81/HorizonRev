"""Normalization helpers for observations."""

from __future__ import annotations


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_norm(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return clip01(value / scale)
