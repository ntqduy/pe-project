from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .base import FusionModule, ordered_features


class ConcatMLPFusion(FusionModule):
    def __init__(
        self,
        feature_dim: int,
        feature_names: Sequence[str],
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("concat fusion requires at least one named input")
        self.output_dim = int(output_dim)
        hidden = int(hidden_dim or output_dim)
        self.network = nn.Sequential(
            nn.Linear(int(feature_dim) * len(self.feature_names), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.output_dim),
        )

    def forward(
        self,
        features: Mapping[str, Tensor],
        availability: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, None]:
        values = ordered_features(self.feature_names, features)
        masked: list[Tensor] = []
        for name, value in zip(self.feature_names, values):
            if availability is not None and name in availability:
                present = availability[name].to(value.device).bool()
                if present.ndim != 1 or present.shape[0] != value.shape[0]:
                    raise ValueError(f"availability for {name} must have shape [B]")
                value = value * present[:, None].to(value.dtype)
            masked.append(value)
        return self.network(torch.cat(masked, dim=1)), None
