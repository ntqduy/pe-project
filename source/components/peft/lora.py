from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, values: Tensor) -> Tensor:
        update = (self.dropout(values) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(values) + update * self.scale


def inject_lora(
    module: nn.Module,
    target_modules: Iterable[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    targets = tuple(target_modules)
    if not targets:
        raise ValueError("LoRA target_modules cannot be empty")
    replaced: list[str] = []
    for full_name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear) or not any(token in full_name for token in targets):
            continue
        parent_name, _, attribute = full_name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attribute, LoRALinear(child, rank, alpha, dropout))
        replaced.append(full_name)
    if not replaced:
        raise ValueError(f"no Linear modules matched LoRA targets: {targets}")
    return replaced
