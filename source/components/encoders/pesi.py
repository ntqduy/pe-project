from __future__ import annotations

import torch
from torch import Tensor, nn


class PESIEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, scores: Tensor, missing: Tensor | None = None) -> Tensor:
        if scores.ndim != 2 or scores.shape[1] != self.input_dim:
            raise ValueError(f"expected PESI/sPESI [B,{self.input_dim}], got {tuple(scores.shape)}")
        inferred = torch.isnan(scores)
        missing_mask = inferred if missing is None else missing.bool() | inferred
        values = torch.nan_to_num(scores, nan=0.0)
        return self.network(torch.cat((values, missing_mask.to(values.dtype)), dim=1))
