from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional

from source.components.targets import masked_target_loss
from source.tasks.diagnosis.organ_targets import OrganTarget


def diagnosis_loss(
    logits: Mapping[str, Tensor],
    labels: Mapping[str, Tensor],
    masks: Mapping[str, Tensor] | None = None,
    weights: Mapping[str, float] | None = None,
    *,
    allow_empty: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    losses: list[Tensor] = []
    report: dict[str, float] = {}
    for target, prediction in logits.items():
        if target not in labels:
            continue
        truth = labels[target].to(prediction.device)
        valid = masks[target].to(prediction.device).bool() if masks and target in masks else ~torch.isnan(truth.float())
        if not valid.any():
            continue
        if prediction.shape[-1] == 1:
            item = functional.binary_cross_entropy_with_logits(
                prediction.squeeze(-1)[valid], truth.float()[valid]
            )
        else:
            item = functional.cross_entropy(prediction[valid], truth.long()[valid])
        weighted = item * float((weights or {}).get(target, 1.0))
        losses.append(weighted)
        report[target] = float(item.detach().cpu())
    if not losses:
        if allow_empty and logits:
            first = next(iter(logits.values()))
            return sum((value.sum() * 0 for value in logits.values()), first.new_zeros(())), report
        raise ValueError("batch contains no valid diagnosis targets")
    return torch.stack(losses).sum(), report


def organ_auxiliary_loss(
    auxiliary_logits: Mapping[str, Mapping[str, Tensor]],
    target_mapping: Mapping[str, Mapping[str, OrganTarget]],
    labels_by_source: Mapping[str, Mapping[str, Tensor]],
    valid_by_source: Mapping[str, Mapping[str, Tensor]],
    *,
    organ_weights: Mapping[str, float] | None = None,
    target_weights: Mapping[str, float] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Masked, branch-specific auxiliary loss with per-target valid counts."""

    differentiable: Tensor | None = None
    organ_losses: list[Tensor] = []
    report: dict[str, float] = {}
    for organ, logits in auxiliary_logits.items():
        if organ not in target_mapping:
            raise KeyError(f"auxiliary logits have no target mapping for organ {organ!r}")
        target_losses: list[Tensor] = []
        for name, prediction in logits.items():
            zero = prediction.sum() * 0
            differentiable = zero if differentiable is None else differentiable + zero
            target = target_mapping[organ][name]
            source_labels = labels_by_source.get(target.source, {})
            source_valid = valid_by_source.get(target.source, {})
            if name not in source_labels:
                raise KeyError(f"missing {target.source} auxiliary labels for {organ}.{name}")
            truth = source_labels[name].to(prediction.device)
            valid = source_valid.get(name, torch.isfinite(truth.float())).to(prediction.device).bool()
            valid = valid & torch.isfinite(truth.float())
            count = int(valid.sum().detach().cpu())
            prefix = f"auxiliary.{organ}.{name}"
            report[f"{prefix}.valid_count"] = float(count)
            if count == 0:
                report[f"{prefix}.loss"] = 0.0
                continue
            item = masked_target_loss(prediction, truth, valid, target.spec).mean()
            weight = float((target_weights or {}).get(name, 1.0))
            if weight < 0:
                raise ValueError(f"auxiliary target weight must be non-negative: {name}={weight}")
            report[f"{prefix}.loss"] = float(item.detach().cpu())
            if weight > 0:
                target_losses.append(item * weight)
        if target_losses:
            organ_loss = torch.stack(target_losses).sum()
            weight = float((organ_weights or {}).get(organ, 1.0))
            if weight < 0:
                raise ValueError(f"auxiliary organ weight must be non-negative: {organ}={weight}")
            report[f"auxiliary.{organ}.loss"] = float(organ_loss.detach().cpu())
            if weight > 0:
                organ_losses.append(organ_loss * weight)
        else:
            report[f"auxiliary.{organ}.loss"] = 0.0
    if organ_losses:
        return torch.stack(organ_losses).sum() + differentiable, report
    if differentiable is None:
        raise ValueError("no auxiliary logits were supplied")
    return differentiable, report
