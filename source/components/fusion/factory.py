"""One place that decides which fusion a config asks for.

Three fusion types are supported, and they are the only difference between the `concat` /
`late` / `moe` cells of the diagnosis and prognosis experiment grids:

    concat_mlp  feature-level: masked concatenation of every branch, then one MLP
    soft_moe    feature-level: a learned per-patient router over the branches
    late_logit  logit-level: one head per branch, then a masked, renormalized weighted
                average of the branch logits

`build_fusion` returns a :class:`FusionModule` for the two feature-level types. `late_logit`
is not a feature-level module -- it combines *decisions*, so it needs the task's prediction
heads and is constructed by the task model itself (see `source/tasks/*/model.py`). Ask
`is_late_logit()` before calling `build_fusion`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import FusionModule
from .concat_mlp import ConcatMLPFusion
from .soft_moe import SoftMoEFusion

CONCAT_MLP = "concat_mlp"
SOFT_MOE = "soft_moe"
LATE_LOGIT = "late_logit"

_CONCAT_ALIASES = {"concat", CONCAT_MLP}
_MOE_ALIASES = {"moe", "dense_soft_moe", SOFT_MOE}
_LATE_ALIASES = {"late", "late_logit", "late_fusion"}


def resolve_fusion_type(
    config: Mapping[str, Any] | None, fallback_type: str = SOFT_MOE
) -> str:
    """Normalize `fusion.type` (falling back to `task.architecture`) to a canonical name."""
    requested = str(dict(config or {}).get("type") or fallback_type).strip().lower()
    if requested in _CONCAT_ALIASES:
        return CONCAT_MLP
    if requested in _MOE_ALIASES:
        return SOFT_MOE
    if requested in _LATE_ALIASES:
        return LATE_LOGIT
    raise ValueError(f"unsupported fusion type: {requested}")


def is_late_logit(value: Any) -> bool:
    """True when a fusion type (or a raw config value) selects logit-level fusion."""
    return str(value or "").strip().lower() in _LATE_ALIASES


def build_fusion(
    config: Mapping[str, Any] | None,
    feature_names: Sequence[str],
    feature_dim: int,
    *,
    output_dim: int | None = None,
    fallback_type: str = SOFT_MOE,
) -> FusionModule:
    options = dict(config or {})
    fusion_type = resolve_fusion_type(options, fallback_type)
    names = tuple(feature_names)
    if fusion_type == CONCAT_MLP:
        return ConcatMLPFusion(
            feature_dim,
            names,
            int(output_dim or feature_dim),
            hidden_dim=int(options["hidden_dim"]) if options.get("hidden_dim") else None,
            dropout=float(options.get("dropout", 0.0)),
        )
    if fusion_type == SOFT_MOE:
        if output_dim is not None and int(output_dim) != int(feature_dim):
            raise ValueError("soft_moe output dimension equals its expert feature dimension")
        return SoftMoEFusion(
            feature_dim,
            names,
            hidden_dim=int(options["hidden_dim"]) if options.get("hidden_dim") else None,
            dropout=float(options.get("dropout", 0.0)),
            temperature=float(options.get("temperature", 1.0)),
        )
    raise ValueError(
        "late_logit fuses branch decisions, not branch features; the task model builds it "
        "from source.components.fusion.late_logit.LateLogitFusion"
    )
