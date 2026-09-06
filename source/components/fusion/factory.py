from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import FusionModule
from .concat_mlp import ConcatMLPFusion
from .soft_moe import SoftMoEFusion


def build_fusion(
    config: Mapping[str, Any] | None,
    feature_names: Sequence[str],
    feature_dim: int,
    *,
    output_dim: int | None = None,
    fallback_type: str = "soft_moe",
) -> FusionModule:
    options = dict(config or {})
    fusion_type = str(options.get("type") or fallback_type).lower()
    names = tuple(feature_names)
    if fusion_type in {"concat", "concat_mlp"}:
        return ConcatMLPFusion(
            feature_dim,
            names,
            int(output_dim or feature_dim),
            hidden_dim=int(options["hidden_dim"]) if options.get("hidden_dim") else None,
            dropout=float(options.get("dropout", 0.0)),
        )
    if fusion_type in {"moe", "dense_soft_moe", "soft_moe"}:
        if output_dim is not None and int(output_dim) != int(feature_dim):
            raise ValueError("soft_moe output dimension equals its expert feature dimension")
        return SoftMoEFusion(
            feature_dim,
            names,
            hidden_dim=int(options["hidden_dim"]) if options.get("hidden_dim") else None,
            dropout=float(options.get("dropout", 0.0)),
            temperature=float(options.get("temperature", 1.0)),
        )
    raise ValueError(f"unsupported fusion type: {fusion_type}")
