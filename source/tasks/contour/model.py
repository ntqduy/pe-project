from __future__ import annotations

from torch import Tensor, nn

from source.components.encoders.image.base import BaseImageEncoder

from .decoder import ContourDecoder


class ContourModel(nn.Module):
    def __init__(self, image_encoder: BaseImageEncoder, regions: int = 1, decoder_channels: int = 64):
        super().__init__()
        self.image_encoder = image_encoder
        self.decoder = ContourDecoder(image_encoder.feature_dim, regions, decoder_channels)

    def forward(self, volume: Tensor) -> Tensor:
        features = self.image_encoder.forward_features(volume)
        return self.decoder(features.feature_map, tuple(volume.shape[-3:]))
