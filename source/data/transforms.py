from __future__ import annotations

from collections.abc import Callable, Sequence

import torch.nn.functional as functional
from torch import Tensor


class Compose:
    def __init__(self, transforms: Sequence[Callable[[Tensor], Tensor]]):
        self.transforms = tuple(transforms)

    def __call__(self, value: Tensor) -> Tensor:
        for transform in self.transforms:
            value = transform(value)
        return value


class CTWindowNormalize:
    def __init__(self, lower: float = -1000.0, upper: float = 1000.0):
        if lower >= upper:
            raise ValueError("CT window lower bound must be less than upper bound")
        self.lower = float(lower)
        self.upper = float(upper)

    def __call__(self, volume: Tensor) -> Tensor:
        volume = volume.float().clamp(self.lower, self.upper)
        return (volume - self.lower) / (self.upper - self.lower) * 2 - 1


class ResizeVolume:
    def __init__(self, shape: tuple[int, int, int], mode: str = "trilinear"):
        self.shape = tuple(int(value) for value in shape)
        self.mode = mode

    def __call__(self, volume: Tensor) -> Tensor:
        needs_batch = volume.ndim == 4
        source = volume.unsqueeze(0) if needs_batch else volume
        if source.ndim != 5:
            raise ValueError(f"expected [C,D,H,W] or [B,C,D,H,W], got {tuple(volume.shape)}")
        options = {"align_corners": False} if self.mode in {"linear", "bilinear", "bicubic", "trilinear"} else {}
        result = functional.interpolate(source.float(), size=self.shape, mode=self.mode, **options)
        return result.squeeze(0) if needs_batch else result
