"""Optional per-epoch learning-rate schedules (``training.scheduler``).

``Trainer`` calls ``scheduler.step(validation_metric)`` once per epoch. A torch LambdaLR
would read that metric as an epoch index, so the schedule is wrapped to ignore it.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


class EpochCosineSchedule:
    """Linear warm-up for ``warmup_epochs``, then cosine decay to ``min_factor`` x base lr."""

    def __init__(self, optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int = 0, min_factor: float = 0.01):
        self.epochs = max(1, int(epochs))
        self.warmup = max(0, int(warmup_epochs))
        self.min_factor = float(min_factor)
        self.inner = torch.optim.lr_scheduler.LambdaLR(optimizer, self.factor)

    def factor(self, epoch: int) -> float:
        # LambdaLR evaluates epoch 0 at construction; epoch e is the lr used during epoch e+1.
        if self.warmup and epoch < self.warmup:
            return (epoch + 1) / (self.warmup + 1)
        span = max(1, self.epochs - self.warmup)
        progress = min(1.0, (epoch - self.warmup) / span)
        return self.min_factor + (1 - self.min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    def step(self, *_: Any) -> None:
        self.inner.step()

    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.inner.load_state_dict(dict(state))


def build_scheduler(optimizer: torch.optim.Optimizer, training: Mapping[str, Any]) -> EpochCosineSchedule | None:
    kind = str(training.get("scheduler") or "none").lower()
    if kind in {"none", "constant", ""}:
        return None
    if kind == "cosine":
        return EpochCosineSchedule(
            optimizer,
            int(training.get("epochs", 1)),
            warmup_epochs=int(training.get("warmup_epochs", 0)),
            min_factor=float(training.get("min_lr_factor", 0.01)),
        )
    raise ValueError(f"training.scheduler must be none or cosine, got {kind!r}")
