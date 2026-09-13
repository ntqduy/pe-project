"""Small end-to-end 3D CNN baseline inspired by the PENet comparison protocol.

This is intentionally a project-native, randomly initialized encoder.  It is not
advertised as a byte-for-byte reproduction of the published PENet implementation;
its purpose is to provide the strong non-foundation 3D baseline requested in the
study email while obeying the same ``BaseImageEncoder`` interface as the FM arms.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .base import BaseImageEncoder, ImageFeatures


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv3d(input_channels, output_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.body(values) + self.skip(values))


class PENetStyleEncoder(BaseImageEncoder):
    """End-to-end 3D residual encoder with no public-FM initialization."""

    def __init__(
        self,
        in_channels: int = 1,
        widths: Sequence[int] = (32, 64, 128, 256),
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        widths = tuple(int(value) for value in widths)
        if not widths or any(value < 1 for value in widths):
            raise ValueError("PENet-style widths must contain positive integers")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be positive")
        self.stem = nn.Sequential(
            nn.Conv3d(int(in_channels), widths[0], 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.GELU(),
        )
        stages: list[nn.Module] = []
        previous = widths[0]
        for stage_index, width in enumerate(widths):
            blocks: list[nn.Module] = []
            for block_index in range(int(blocks_per_stage)):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(ResidualBlock3D(previous, width, stride=stride))
                previous = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.feature_dim = widths[-1]

    def forward_features(self, volume: Tensor) -> ImageFeatures:
        feature = self.stem(volume)
        pyramid: list[Tensor] = []
        for stage in self.stages:
            feature = stage(feature)
            pyramid.append(feature)
        embedding = feature.mean(dim=(2, 3, 4))
        return ImageFeatures(
            feature_map=feature,
            global_embedding=embedding,
            pyramid=tuple(pyramid),
            metadata={"backbone": "penet_style", "initialization": "random"},
        )

    def load_pretrained_weights(self, checkpoint: str, strict: bool = True) -> dict[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping):
            for key in ("model_state", "state_dict", "model"):
                if key in payload and isinstance(payload[key], Mapping):
                    payload = payload[key]
                    break
        if not isinstance(payload, Mapping):
            raise ValueError("PENet-style checkpoint does not contain a state dictionary")
        incompatible = self.load_state_dict(payload, strict=strict)
        return {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }


def build_penet_style(config: Mapping[str, Any]) -> PENetStyleEncoder:
    return PENetStyleEncoder(
        in_channels=int(config.get("in_channels", 1)),
        widths=tuple(config.get("widths") or (32, 64, 128, 256)),
        blocks_per_stage=int(config.get("blocks_per_stage", 2)),
    )


__all__ = ["PENetStyleEncoder", "ResidualBlock3D", "build_penet_style"]
