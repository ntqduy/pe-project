"""Whole-volume baseline classifier: encoder -> (standardize) -> shared projection -> head.

    volume [B,1,x,y,z] -> encoder.global_embedding [B, F]
                       -> optional per-channel z-score fit on train (frozen encoders)
                       -> projection  Dropout -> Linear(F, P) -> LayerNorm(P)    (shared)
                       -> head        MLP or KAN : P -> sum of target output dims
                       -> one logit tensor per configured target

The projection is identical for both heads, so the head ablation changes only the head.
Outputs follow the diagnosis / prognosis model contracts, so the task losses, trainer,
evaluator and Grad-CAM preview are reused unchanged:

    diagnosis  {"logits": {target: [B, dim]}, ...}
    prognosis  {"logits": [B] (primary), "target_logits": {target: [B]}, ...}
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from source.components.adapters.standardization import PooledFeatureStandardizer
from source.components.encoders.image.base import BaseImageEncoder
from source.components.targets import target_output_dims

from .head import build_head


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: BaseImageEncoder,
        stage: str,
        targets: Mapping[str, Any],
        *,
        primary_target: str,
        head: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if stage not in {"diagnosis", "prognosis"}:
            raise ValueError(f"baseline classifier supports diagnosis/prognosis, got {stage!r}")
        options = dict(head or {})
        self.stage = stage
        self.image_encoder = image_encoder
        self.targets = {str(name): value for name, value in dict(targets).items()}
        self.primary_target = str(primary_target)
        if self.primary_target not in self.targets:
            raise ValueError(f"primary target {self.primary_target!r} is not among {sorted(self.targets)}")
        self.output_dims = target_output_dims(self.targets)
        if stage == "prognosis" and any(dim != 1 for dim in self.output_dims.values()):
            raise ValueError("prognosis baseline targets must be binary")
        feature_dim = int(image_encoder.feature_dim)
        projection_dim = int(options.get("projection_dim", 64))
        self.standardizer = (
            PooledFeatureStandardizer(("global",), feature_dim)
            if bool(options.get("standardize_inputs", False))
            else None
        )
        self.projection = nn.Sequential(
            nn.Dropout(float(options.get("projection_dropout", 0.0))),
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.head_type = str(options.get("type", "mlp")).lower()
        self.head = build_head(self.head_type, projection_dim, sum(self.output_dims.values()), options)

    # ---------------------------------------------------------------- forward
    def embed(self, volume: Tensor) -> Tensor:
        features = self.image_encoder.forward_features(volume)
        return features.global_embedding.float()

    def forward(self, inputs: Any, masks: Mapping[str, Tensor] | None = None) -> dict[str, Any]:
        volume = inputs["volume"] if isinstance(inputs, Mapping) else inputs
        embedding = self.embed(volume)
        if self.standardizer is not None:
            embedding = self.standardizer({"global": embedding})["global"]
        projected = self.projection(embedding)
        stacked = self.head(projected)
        logits: dict[str, Tensor] = {}
        start = 0
        for name, dim in self.output_dims.items():
            logits[name] = stacked[:, start : start + dim]
            start += dim
        common = {"features": projected, "routing": None, "roi_present": {}, "branch_features": {}}
        if self.stage == "diagnosis":
            return {"logits": logits, "auxiliary_logits": {}, "auxiliary_target_mapping": {}, **common}
        squeezed = {name: value.squeeze(-1) for name, value in logits.items()}
        return {
            "logits": squeezed[self.primary_target],
            "target_logits": squeezed,
            "primary_target": self.primary_target,
            **common,
        }

    # ---------------------------------------------------------------- train-split standardization
    @torch.no_grad()
    def fit_input_standardizer(self, loader: Any, device: torch.device, distributed: bool = False) -> dict[str, Any] | None:
        """Fit the pooled-feature z-score on the (sharded) train loader; all-reduce across ranks."""
        if self.standardizer is None:
            return None
        was_training = self.training
        self.to(device).eval()
        total = torch.zeros(1, self.standardizer.feature_dim, dtype=torch.float64, device=device)
        total_square = torch.zeros_like(total)
        count = torch.zeros(1, dtype=torch.float64, device=device)
        started = time.perf_counter()
        for batch in loader:
            volume = batch["volume"].to(device)
            values = self.embed(volume).to(torch.float64)
            total[0] += values.sum(dim=0)
            total_square[0] += values.square().sum(dim=0)
            count[0] += values.shape[0]
        if distributed:
            import torch.distributed as dist

            for tensor in (total, total_square, count):
                dist.all_reduce(tensor)
        identity = self.standardizer.fit_moments(total, total_square, count)
        self.train(was_training)
        return {
            "branches": ["global"],
            "rows": int(count.item()),
            "identity_branches": list(identity),
            "fit_seconds": round(time.perf_counter() - started, 1),
        }


__all__ = ["BaselineClassifier"]
