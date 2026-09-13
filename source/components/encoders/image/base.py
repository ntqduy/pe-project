from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn


@dataclass
class ImageFeatures:
    feature_map: Tensor
    global_embedding: Tensor
    pyramid: tuple[Tensor, ...] = ()
    metadata: dict[str, Any] | None = None


class BaseImageEncoder(nn.Module, ABC):
    """Stable project contract around inspected third-party CT encoders."""

    feature_dim: int

    @abstractmethod
    def forward_features(self, volume: Tensor) -> ImageFeatures:
        raise NotImplementedError

    def forward(self, volume: Tensor) -> Tensor:
        return self.forward_features(volume).global_embedding

    def get_feature_map(self, volume: Tensor) -> Tensor:
        return self.forward_features(volume).feature_map

    def get_global_embedding(self, volume: Tensor) -> Tensor:
        return self.forward_features(volume).global_embedding

    @abstractmethod
    def load_pretrained_weights(self, checkpoint: str, strict: bool = True) -> dict[str, Any]:
        raise NotImplementedError
