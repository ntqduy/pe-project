from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from source.components.targets import target_output_dims


class DiagnosisHeads(nn.Module):
    def __init__(self, input_dim: int, targets: Mapping[str, Any]):
        super().__init__()
        dimensions = target_output_dims(targets)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(input_dim, output_dim) for name, output_dim in dimensions.items()}
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {name: head(features) for name, head in self.heads.items()}
