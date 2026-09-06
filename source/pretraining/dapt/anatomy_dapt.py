from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn

from source.components.roi.masks import keep_only_roi

from .base import DAPTObjective


class AnatomyAwareDAPT(DAPTObjective):
    def __init__(self, encoder, projection_dim: int = 128):
        super().__init__(encoder)
        self.projector = nn.Linear(encoder.feature_dim, projection_dim)

    def forward(self, volume: Tensor, anatomy_mask: Tensor) -> dict[str, Tensor]:
        global_embedding = functional.normalize(self.projector(self.image_encoder.get_global_embedding(volume)), dim=1)
        anatomy_volume = keep_only_roi(volume, anatomy_mask)
        anatomy_embedding = functional.normalize(self.projector(self.image_encoder.get_global_embedding(anatomy_volume)), dim=1)
        loss = (1 - (global_embedding * anatomy_embedding).sum(dim=1)).mean()
        return {"loss": loss, "global_embedding": global_embedding, "anatomy_embedding": anatomy_embedding}
