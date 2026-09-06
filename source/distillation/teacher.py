from __future__ import annotations

from typing import Any

import torch
from torch import nn


def freeze_teacher(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


class FrozenTeacher(nn.Module):
    """Read-only task teacher; weights never enter the student's optimizer/state."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = freeze_teacher(model)

    def train(self, mode: bool = True) -> "FrozenTeacher":
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
