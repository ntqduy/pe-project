from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from source.components.encoders.image.base import BaseImageEncoder


class ImageReportAlignment(nn.Module):
    def __init__(
        self,
        image_encoder: BaseImageEncoder,
        report_encoder: nn.Module,
        report_dim: int,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.report_encoder = report_encoder
        self.image_projection = nn.Linear(image_encoder.feature_dim, projection_dim)
        self.report_projection = nn.Linear(report_dim, projection_dim)
        self.temperature = temperature

    def forward(self, volume: Tensor, report_inputs: object) -> dict[str, Tensor]:
        image = functional.normalize(self.image_projection(self.image_encoder.get_global_embedding(volume)), dim=1)
        report_output = self.report_encoder(report_inputs)
        if not isinstance(report_output, Tensor):
            if hasattr(report_output, "pooler_output"):
                report_output = report_output.pooler_output
            elif hasattr(report_output, "last_hidden_state"):
                report_output = report_output.last_hidden_state[:, 0]
            else:
                raise TypeError("inspected report adapter must return a tensor or standard pooled output")
        report = functional.normalize(self.report_projection(report_output), dim=1)
        logits = image @ report.t() / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss = (functional.cross_entropy(logits, targets) + functional.cross_entropy(logits.t(), targets)) / 2
        return {"loss": loss, "image_embedding": image, "report_embedding": report}
