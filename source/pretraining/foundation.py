from __future__ import annotations

from torch import nn

from source.components.encoders.image.base import BaseImageEncoder


class FoundationModel(nn.Module):
    """Checkpoint namespace shared by public and project-adapted image encoders."""

    def __init__(self, image_encoder: BaseImageEncoder):
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, volume):
        return self.image_encoder.forward_features(volume)
