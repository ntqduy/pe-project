from __future__ import annotations

from torch import Tensor

from .base import DAPTObjective


class NoDAPT(DAPTObjective):
    def forward(self, volume: Tensor) -> dict[str, Tensor]:
        return {"embedding": self.image_encoder.get_global_embedding(volume)}
