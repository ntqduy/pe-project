"""Masked averaging of named branch logits, independent of feature fusion."""
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


class LateLogitFusion(nn.Module):
    def __init__(self, branch_names: Sequence[str], *, learned: bool = False,
                 weights: Mapping[str, float] | None = None):
        super().__init__()
        self.branch_names = tuple(branch_names)
        if not self.branch_names or len(set(self.branch_names)) != len(self.branch_names):
            raise ValueError("late fusion requires unique branch names")
        if weights is not None and set(weights) != set(self.branch_names):
            raise ValueError("fixed weights must name every branch exactly once")
        values = torch.tensor([float((weights or {}).get(n, 1.0)) for n in self.branch_names])
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("branch weights must be finite and positive")
        self.learned = learned
        if learned:
            self.log_weights = nn.Parameter(values.log())
        else:
            self.register_buffer("fixed_weights", values)

    def forward(self, logits: Mapping[str, Tensor], availability: Mapping[str, Tensor] | None = None):
        if not logits or set(logits) - set(self.branch_names):
            raise ValueError("provide at least one known branch")
        reference = next(iter(logits.values()))
        if reference.ndim not in (1, 2):
            raise ValueError("branch logits must have shape [B] or [B,C]")
        values, masks = [], []
        for name in self.branch_names:
            value = logits.get(name, torch.zeros_like(reference))
            if value.shape != reference.shape:
                raise ValueError("branch logits must have identical shapes")
            present = torch.full((reference.shape[0],), name in logits, device=reference.device, dtype=torch.bool)
            if availability is not None and name in availability:
                mask = availability[name].to(device=reference.device, dtype=torch.bool)
                if mask.shape != present.shape:
                    raise ValueError("branch availability must have shape [B]")
                present = present & mask
            masks.append(present)
            values.append(torch.where(present.reshape((-1,) + (1,) * (value.ndim - 1)), value, 0.0))
        present = torch.stack(masks, dim=1)
        if not present.any(dim=1).all():
            raise ValueError("late fusion requires an available branch for every patient")
        if self.learned:
            scores = self.log_weights.expand(reference.shape[0], -1).masked_fill(~present, -torch.inf)
            weights = scores.softmax(dim=1)
        else:
            weights = self.fixed_weights[None, :] * present
            weights = weights / weights.sum(dim=1, keepdim=True)
        weights = weights.to(reference.dtype)
        stacked = torch.stack(values, dim=1)
        broadcast = weights if reference.ndim == 1 else weights.unsqueeze(-1)
        return (stacked * broadcast).sum(dim=1), weights
