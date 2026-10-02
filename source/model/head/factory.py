from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

HEAD_TYPES = ("mlp", "kan")


def build_head(kind: str, input_dim: int, output_dim: int, config: Mapping[str, Any] | None = None) -> nn.Module:
    """``head.type`` -> module mapping the shared projection to the stacked target logits."""
    options = dict(config or {})
    normalized = str(kind).strip().lower()
    if normalized == "mlp":
        from .mlp import MLPHead

        mlp = dict(options.get("mlp") or {})
        return MLPHead(
            input_dim,
            output_dim,
            hidden_dim=int(mlp.get("hidden_dim", 64)),
            dropout=float(mlp.get("dropout", 0.1)),
        )
    if normalized == "kan":
        from .kan import KANHead

        kan = dict(options.get("kan") or {})
        return KANHead(
            input_dim,
            output_dim,
            hidden_dims=tuple(kan.get("hidden_dims") or (16,)),
            grid=int(kan.get("grid", 5)),
            spline_order=int(kan.get("spline_order", 3)),
            grid_range=tuple(kan.get("grid_range") or (-3.0, 3.0)),
            seed=int(kan.get("seed", 42)),
        )
    raise ValueError(f"unknown head.type {kind!r}; choose one of {HEAD_TYPES}")
