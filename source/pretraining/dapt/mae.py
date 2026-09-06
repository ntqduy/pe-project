from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .base import DAPTObjective


class MaskedAutoencoderDAPT(DAPTObjective):
    def __init__(self, encoder, mask_ratio: float = 0.75):
        super().__init__(encoder)
        if not 0 < mask_ratio < 1:
            raise ValueError("MAE mask_ratio must be between zero and one")
        self.mask_ratio = mask_ratio
        self.decoder = nn.Conv3d(encoder.feature_dim, 1, 1)

    def forward(self, volume: Tensor) -> dict[str, Tensor]:
        mask = torch.rand_like(volume[:, :1]) < self.mask_ratio
        masked = volume.masked_fill(mask, 0)
        features = self.image_encoder.forward_features(masked).feature_map
        reconstruction = functional.interpolate(self.decoder(features), size=volume.shape[-3:], mode="trilinear", align_corners=False)
        loss = ((reconstruction - volume[:, :1]).square() * mask).sum() / mask.sum().clamp_min(1)
        return {"loss": loss, "reconstruction": reconstruction, "mask": mask}
