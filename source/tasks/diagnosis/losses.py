from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional



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
