from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import Tensor
from torch.nn import functional


TargetKind = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class TargetSpec:
    """Typed supervision contract shared by diagnosis, concepts, and silver adaptation."""

    kind: TargetKind
    classes: int = 1
    loss: str | None = None

    @property
    def output_dim(self) -> int:
        return self.classes if self.kind == "multiclass" else 1

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "classes": self.classes, "loss": self.loss}


def normalize_target_spec(value: Any) -> TargetSpec:
    if isinstance(value, TargetSpec):
        return value
    if isinstance(value, int):
        if value < 1:
            raise ValueError("target class count must be positive")
        return TargetSpec("binary" if value == 1 else "multiclass", classes=value)
    if isinstance(value, str):
        value = {"type": value}
    if not isinstance(value, Mapping):
        raise TypeError(f"target specification must be an integer or mapping, got {type(value).__name__}")
    kind = str(value.get("type") or value.get("kind") or "").strip().lower()
    aliases = {"continuous": "regression", "categorical": "multiclass"}
    kind = aliases.get(kind, kind)
    if kind not in {"binary", "multiclass", "regression"}:
        raise ValueError(f"unsupported target type: {kind!r}")
    default_classes = 2 if kind == "multiclass" else 1
    classes = int(value.get("classes", value.get("num_classes", default_classes)))
    if kind == "multiclass" and classes < 2:
        raise ValueError("multiclass targets require at least two classes")
    if kind != "multiclass" and classes != 1:
        raise ValueError(f"{kind} targets must use classes=1")
    loss = value.get("loss")
    return TargetSpec(kind, classes=classes, loss=str(loss).lower() if loss else None)


def normalize_target_specs(targets: Mapping[str, Any]) -> dict[str, TargetSpec]:
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("at least one target specification is required")
    return {str(name): normalize_target_spec(value) for name, value in targets.items()}


def target_output_dims(targets: Mapping[str, Any]) -> dict[str, int]:
    return {name: spec.output_dim for name, spec in normalize_target_specs(targets).items()}


def differentiable_zero(logits: Mapping[str, Tensor]) -> Tensor:
    values = list(logits.values())
    if not values:
        raise ValueError("cannot construct a differentiable zero without logits")
    return sum((value.sum() * 0 for value in values), values[0].new_zeros(()))


def masked_target_loss(prediction: Tensor, truth: Tensor, valid: Tensor, spec: TargetSpec) -> Tensor:
    selected = valid.bool() & torch.isfinite(truth.float())
    if not bool(selected.any()):
        return prediction.sum() * 0
    if spec.kind == "binary":
        values = prediction.squeeze(-1)[selected]
        return functional.binary_cross_entropy_with_logits(values, truth.float()[selected], reduction="none")
    if spec.kind == "multiclass":
        return functional.cross_entropy(prediction[selected], truth.long()[selected], reduction="none")
    values = prediction.squeeze(-1)[selected]
    target = truth.float()[selected]
    if spec.loss in {None, "smooth_l1", "huber"}:
        return functional.smooth_l1_loss(values, target, reduction="none")
    if spec.loss in {"mse", "l2"}:
        return functional.mse_loss(values, target, reduction="none")
    if spec.loss in {"mae", "l1"}:
        return functional.l1_loss(values, target, reduction="none")
    raise ValueError(f"unsupported regression loss: {spec.loss}")


def masked_multitask_loss(
    logits: Mapping[str, Tensor],
    labels: Mapping[str, Tensor],
    valid: Mapping[str, Tensor] | None,
    target_specs: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    *,
    allow_empty: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    specs = normalize_target_specs(target_specs)
    losses: list[Tensor] = []
    report: dict[str, float] = {}
    for name, prediction in logits.items():
        if name not in labels or name not in specs:
            continue
        truth = labels[name].to(prediction.device)
        selected = (
            valid[name].to(prediction.device).bool()
            if valid is not None and name in valid
            else torch.isfinite(truth.float())
        )
        selected = selected & torch.isfinite(truth.float())
        if not bool(selected.any()):
            continue
        weight = float((weights or {}).get(name, 1.0))
        if weight < 0:
            raise ValueError(f"target weight must be non-negative: {name}={weight}")
        if weight == 0:
            continue
        item_values = masked_target_loss(prediction, truth, selected, specs[name])
        item = item_values.mean()
        losses.append(item * weight)
        report[name] = float(item.detach().cpu())
    if not losses:
        if allow_empty:
            return differentiable_zero(logits), report
        raise ValueError("batch contains no valid targets with positive weight")
    return torch.stack(losses).sum(), report
