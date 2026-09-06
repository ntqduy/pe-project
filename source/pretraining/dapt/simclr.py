from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .base import DAPTObjective


class SimCLRDAPT(DAPTObjective):
    def __init__(self, encoder, projection_dim: int = 128, temperature: float = 0.1):
        super().__init__(encoder)
        self.projector = nn.Sequential(nn.Linear(encoder.feature_dim, encoder.feature_dim), nn.GELU(), nn.Linear(encoder.feature_dim, projection_dim))
        self.temperature = temperature

    def forward(self, view_a: Tensor, view_b: Tensor) -> dict[str, Tensor]:
        first = functional.normalize(self.projector(self.image_encoder.get_global_embedding(view_a)), dim=1)
        second = functional.normalize(self.projector(self.image_encoder.get_global_embedding(view_b)), dim=1)
        embeddings = torch.cat((first, second), dim=0)
        similarity = embeddings @ embeddings.t() / self.temperature
        similarity.fill_diagonal_(torch.finfo(similarity.dtype).min)
        batch = first.shape[0]
        targets = torch.cat((torch.arange(batch, 2 * batch, device=first.device), torch.arange(batch, device=first.device)))
        return {"loss": functional.cross_entropy(similarity, targets), "embedding_a": first, "embedding_b": second}
