"""Space helpers compatible with OpenEnv-style environments."""

from __future__ import annotations

import numpy as np

try:
    from openenv import spaces as openenv_spaces  # type: ignore
except Exception:  # pragma: no cover - fallback when OpenEnv layout differs
    openenv_spaces = None


class _FallbackDiscrete:
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def sample(self, rng: np.random.Generator | None = None) -> int:
        generator = rng if rng is not None else np.random.default_rng()
        return int(generator.integers(0, self.n))

    def contains(self, x: int) -> bool:
        return 0 <= int(x) < self.n


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def contains(self, x: np.ndarray) -> bool:
        if x.shape != self.shape:
            return False
        if x.dtype != self.dtype:
            return False
        return bool(np.all(x >= self.low) and np.all(x <= self.high))


def make_discrete(n: int):
    if openenv_spaces is not None and hasattr(openenv_spaces, "Discrete"):
        return openenv_spaces.Discrete(n)
    return _FallbackDiscrete(n)


def make_box(low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
    if openenv_spaces is not None and hasattr(openenv_spaces, "Box"):
        return openenv_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _FallbackBox(low=low, high=high, shape=shape, dtype=dtype)
