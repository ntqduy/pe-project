from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .base import FusionModule, ordered_features


class SoftMoEFusion(FusionModule):
    def __init__(
        self,
        feature_dim: int,
        feature_names: Sequence[str],
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.feature_names = tuple(feature_names)
        if not self.feature_names or temperature <= 0:
            raise ValueError("soft MoE needs named experts and positive temperature")
        self.output_dim = int(feature_dim)
        hidden = int(hidden_dim or feature_dim)
        self.router = nn.Sequential(
            nn.Linear(self.output_dim * len(self.feature_names), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(self.feature_names)),
        )
        self.temperature = float(temperature)

    def forward(
        self,
        features: Mapping[str, Tensor],
        availability: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        values = ordered_features(self.feature_names, features)
        stacked = torch.stack(values, dim=1)
        logits = self.router(torch.cat(values, dim=1)) / self.temperature
        if availability is not None:
            masks = []
            for name, value in zip(self.feature_names, values):
                present = availability.get(name)
                if present is None:
                    present = torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
                present = present.to(value.device).bool()
                if present.ndim != 1 or present.shape[0] != value.shape[0]:
                    raise ValueError(f"availability for {name} must have shape [B]")
                masks.append(present)
            available = torch.stack(masks, dim=1)
            if bool((~available).all(dim=1).any()):
                raise ValueError("each sample needs at least one available fusion input")
            logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1), weights
