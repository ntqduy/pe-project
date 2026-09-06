from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


class BottleneckMLPAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class ResidualAdapter(nn.Module):
    """Residual bottleneck with an explicit projection when dimensions differ."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.down = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values: Tensor) -> Tensor:
        update = self.up(self.dropout(torch.nn.functional.gelu(self.norm(self.down(values)))))
        return self.skip(values) + update


class LoRAFeatureAdapter(nn.Module):
    """Backbone-neutral low-rank adapter over pooled regional features.

    Encoder-internal LoRA remains isolated in ``source.components.peft`` because its
    insertion points are backbone-specific. This adapter provides the same scientific
    low-rank ablation without teaching the generic encoder about a concrete backbone.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA adapter rank must be positive")
        self.output_dim = int(output_dim)
        self.base = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.lora_a = nn.Parameter(torch.empty(rank, input_dim))
        self.lora_b = nn.Parameter(torch.zeros(output_dim, rank))
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, values: Tensor) -> Tensor:
        update = (self.dropout(values) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(values) + update * self.scale


def build_organ_adapter(
    adapter_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    dropout: float = 0.1,
    rank: int = 8,
    alpha: float = 16.0,
) -> nn.Module:
    normalized = adapter_type.strip().lower()
    if normalized in {"bottleneck", "bottleneck_mlp", "mlp"}:
        return BottleneckMLPAdapter(input_dim, hidden_dim, output_dim, dropout)
    if normalized in {"residual", "residual_adapter"}:
        return ResidualAdapter(input_dim, hidden_dim, output_dim, dropout)
    if normalized == "lora":
        return LoRAFeatureAdapter(input_dim, output_dim, rank=rank, alpha=alpha, dropout=dropout)
    raise ValueError(f"unsupported organ adapter type: {adapter_type}")


class OrganAdapterBank(nn.Module):
    """Independent global/heart/PA/lung adapters over a shared encoder feature map."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        regions: Sequence[str] = ("heart", "pa", "lung"),
        *,
        include_global: bool = True,
        adapter_type: str = "bottleneck_mlp",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        rank: int = 8,
        alpha: float = 16.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.regions = tuple(dict.fromkeys(str(region) for region in regions))
        self.include_global = bool(include_global)
        self.enabled = bool(enabled)
        self.output_dim = int(output_dim if enabled else input_dim)
        names = (("global",) if self.include_global else ()) + self.regions
        if not names:
            raise ValueError("organ adapter bank requires global or at least one regional feature")
        self.feature_names = names
        self.adapters = nn.ModuleDict(
            {
                name: (
                    build_organ_adapter(
                        adapter_type,
                        input_dim,
                        hidden_dim,
                        output_dim,
                        dropout=dropout,
                        rank=rank,
                        alpha=alpha,
                    )
                    if enabled
                    else nn.Identity()
                )
                for name in names
            }
        )

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        output_dim: int,
        regions: Sequence[str],
        config: Mapping[str, Any] | None,
        *,
        default_hidden_dim: int = 256,
        default_include_global: bool = True,
    ) -> "OrganAdapterBank":
        options = dict(config or {})
        return cls(
            input_dim,
            output_dim,
            regions,
            include_global=bool(options.get("include_global", default_include_global)),
            adapter_type=str(options.get("type", "bottleneck_mlp")),
            hidden_dim=int(options.get("hidden_dim", default_hidden_dim)),
            dropout=float(options.get("dropout", 0.1)),
            rank=int(options.get("rank", 8)),
            alpha=float(options.get("alpha", 16.0)),
            enabled=bool(options.get("enabled", True)),
        )

    def forward(
        self,
        features: Mapping[str, Tensor],
        availability: Mapping[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise KeyError(f"missing organ features: {missing}")
        adapted: dict[str, Tensor] = {}
        present: dict[str, Tensor] = {}
        for name in self.feature_names:
            values = self.adapters[name](features[name])
            mask = (
                availability[name].to(values.device).bool()
                if availability is not None and name in availability
                else torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
            )
            if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
                raise ValueError(f"availability for {name} must have shape [B]")
            adapted[name] = values * mask[:, None].to(values.dtype)
            present[name] = mask
        return adapted, present
