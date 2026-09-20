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

    def train(self, mode: bool = True):
        """Keep a gradient-frozen encoder in eval mode during task training.

        ``Trainer`` correctly calls ``model.train()`` each epoch, but that recursive
        call would otherwise put a frozen third-party encoder back into train mode.  A
        frozen CT-FM must not update BatchNorm statistics or activate Dropout; only its
        downstream adapters and MLP head are trainable.
        """
        super().train(mode)
        parameters = tuple(self.parameters())
        if mode and parameters and not any(parameter.requires_grad for parameter in parameters):
            super().train(False)
        return self

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
