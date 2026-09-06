from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn


class ContourDecoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, 3, padding=1),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_channels, output_channels, 1),
        )

    def forward(self, features: Tensor, output_shape: tuple[int, int, int]) -> Tensor:
        logits = self.network(features)
        return functional.interpolate(logits, size=output_shape, mode="trilinear", align_corners=False)
