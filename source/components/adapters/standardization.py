from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


class PooledFeatureStandardizer(nn.Module):
    """Per-branch, per-channel z-score of pooled encoder features, fit on training rows only.

    A frozen encoder's pooled features share a large per-channel offset across patients (for
    cached CT-FM ~99% of each global vector is the cohort average), which hides the
    between-patient differences from a small trainable head. ``center``/``scale`` are fit once
    on the train split (tools/tasks/train_task.py) and stored as checkpoint buffers, so
    validation, test and every later reload apply exactly the same shift and scale.
    """

    def __init__(self, feature_names: Sequence[str], feature_dim: int):
        super().__init__()
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("feature standardization needs at least one branch")
        self.feature_dim = int(feature_dim)
        branches = len(self.feature_names)
        self.register_buffer("center", torch.zeros(branches, self.feature_dim))
        self.register_buffer("scale", torch.ones(branches, self.feature_dim))
        self.register_buffer("observed_count", torch.zeros(branches, dtype=torch.long))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    # A standard deviation needs at least two rows; a branch seen in fewer training rows (the
    # PE-positive smoke_30 cohort has one training patient) passes through unstandardized.
    MIN_ROWS = 2

    @torch.no_grad()
    def fit_moments(self, total: Tensor, total_square: Tensor, count: Tensor) -> tuple[str, ...]:
        """Set centre/scale from per-branch sums over training rows.

        ``total``/``total_square`` are [branches, feature_dim] sums of x and x^2, ``count`` the
        [branches] number of rows in which that branch was present. Returns the branches left
        as identity (centre 0, scale 1) because fewer than ``MIN_ROWS`` rows had them.
        """
        expected = (len(self.feature_names), self.feature_dim)
        if tuple(total.shape) != expected or tuple(total_square.shape) != expected:
            raise ValueError(f"feature moments must have shape {expected}")
        count = count.to(torch.float64).reshape(-1)
        usable = (count >= self.MIN_ROWS)[:, None]
        rows = count.clamp_min(1.0)[:, None]
        mean = total.to(torch.float64) / rows
        scale = (total_square.to(torch.float64) / rows - mean.square()).clamp_min(0.0).sqrt()
        # A channel constant over the training rows has no scale to divide by; leave it
        # unscaled rather than blow up a later non-constant value by ~1/0.
        scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
        self.center.copy_(torch.where(usable, mean, torch.zeros_like(mean)).to(self.center.dtype))
        self.scale.copy_(torch.where(usable, scale, torch.ones_like(scale)).to(self.scale.dtype))
        self.observed_count.copy_(count.round().to(torch.long))
        self.fitted.fill_(True)
        return self.identity_branches()

    def identity_branches(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, rows in zip(self.feature_names, self.observed_count.tolist())
            if rows < self.MIN_ROWS
        )

    def forward(self, features: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if not bool(self.fitted.item()):
            raise RuntimeError("pooled feature standardization has not been fit on training data")
        standardized = dict(features)
        for index, name in enumerate(self.feature_names):
            value = standardized.get(name)
            if value is None:
                continue
            center = self.center[index].to(value.device, value.dtype)
            scale = self.scale[index].to(value.device, value.dtype)
            standardized[name] = (value - center) / scale
        return standardized

    def export_state(self) -> dict[str, Any]:
        """Compact summary for logs/result.json; the full vectors live in the checkpoint."""
        return {
            "fit_split": "train",
            "fitted": bool(self.fitted.item()),
            "identity_branches": list(self.identity_branches()),
            "branches": {
                name: {
                    "rows": int(self.observed_count[index].item()),
                    "center_abs_median": float(self.center[index].abs().median().item()),
                    "scale_median": float(self.scale[index].median().item()),
                }
                for index, name in enumerate(self.feature_names)
            },
        }


__all__ = ["PooledFeatureStandardizer"]
