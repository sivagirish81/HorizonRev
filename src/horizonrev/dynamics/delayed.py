"""Delayed-effect queue for churn deltas."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Tuple


DelayedQueue = DefaultDict[int, List[Tuple[float, str]]]


def make_queue() -> DelayedQueue:
    return defaultdict(list)


def schedule_effect(queue: DelayedQueue, target_month: int, delta: float, label: str) -> None:
    queue[target_month].append((float(delta), label))


def pop_effects(queue: DelayedQueue, month: int) -> tuple[float, list[str]]:
    effects = queue.pop(month, [])
    total_delta = float(sum(delta for delta, _ in effects))
    labels = [name for _, name in effects]
    return total_delta, labels
