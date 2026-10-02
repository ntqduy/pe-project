from __future__ import annotations

from torch import Tensor, nn


class MLPHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear on the shared projection."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)
