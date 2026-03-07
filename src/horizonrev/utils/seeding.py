"""Randomness helpers."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)
