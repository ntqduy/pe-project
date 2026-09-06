from __future__ import annotations

from torch import Tensor, nn


class PrognosisHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, values: Tensor) -> Tensor:
        return super().forward(values).squeeze(-1)
