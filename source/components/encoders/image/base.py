from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

# Norm layers that update buffers in train mode (InstanceNorm only with track_running_stats).
_RUNNING_STAT_NORMS = (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm)


@dataclass
class ImageFeatures:
    feature_map: Tensor
    global_embedding: Tensor
    pyramid: tuple[Tensor, ...] = ()
    metadata: dict[str, Any] | None = None


class BaseImageEncoder(nn.Module, ABC):
    """Stable project contract around inspected third-party CT encoders."""

    feature_dim: int

    @property
    def feature_map_dim(self) -> int:
        """Channels of ``feature_map`` (what ROI / anatomy pooling sees).

        ``feature_dim`` is the width of ``global_embedding``. The two differ only for an
        encoder whose embedding is not the spatial mean of its map (nnMamba: concat of the
        c2, c3, c4 means = 14C, map = c4 = 8C), which then sets this explicitly.
        """
        value = self.__dict__.get("_feature_map_dim")
        return int(value) if value else int(self.feature_dim)

    @feature_map_dim.setter
    def feature_map_dim(self, value: int) -> None:
        self.__dict__["_feature_map_dim"] = int(value)

    def train(self, mode: bool = True):
        """Keep a gradient-frozen encoder in eval mode during task training.

        ``Trainer`` correctly calls ``model.train()`` each epoch, but that recursive
        call would otherwise put a frozen third-party encoder back into train mode.  A
        frozen CT-FM must not update BatchNorm statistics or activate Dropout; only its
        downstream adapters and MLP head are trainable.

        A partially frozen encoder (LoRA: frozen base, trainable adapters) stays in train
        mode for its adapters, but every fully frozen submodule (stem, untargeted stages,
        their Dropout/DropPath) and every running-statistics norm layer whose own affine
        parameters are frozen or absent is put back in eval: batch-size-1 updates would
        otherwise overwrite the pretrained BatchNorm statistics. A fully trainable encoder
        (``full`` / scratch) has no frozen parameter and is left entirely in train mode.
        """
        super().train(mode)
        if not mode:
            return self
        parameters = tuple(self.parameters())
        if not parameters or all(parameter.requires_grad for parameter in parameters):
            return self
        if not any(parameter.requires_grad for parameter in parameters):
            return super().train(False)
        for module in self.modules():
            if module is self or not module.training:
                continue
            own = tuple(module.parameters())
            if own and not any(parameter.requires_grad for parameter in own):
                module.train(False)
            elif isinstance(module, _RUNNING_STAT_NORMS) and getattr(module, "track_running_stats", False):
                if not any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
                    module.train(False)
        return self

    @abstractmethod
    def forward_features(self, volume: Tensor) -> ImageFeatures:
        raise NotImplementedError

    def forward(self, volume: Tensor) -> Tensor:
        return self.forward_features(volume).global_embedding

    @abstractmethod
    def load_pretrained_weights(self, checkpoint: str, strict: bool = True) -> dict[str, Any]:
        raise NotImplementedError
