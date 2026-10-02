"""ResNet-18 slice backbone for the 2D / 2.5D slice-MIL baselines."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from source.model.base import BaselineEncoder

from .builder import build_slice_mil_encoder

DEFAULT_TIMM_NAME = "resnet18.a1_in1k"
# timm ResNet: the 3x3 / 1x1 convs of the two deepest stages (as resnet18_3d).
LORA_TARGETS = ("layer3", "layer4")


def build_resnet18_2d(config: Mapping[str, Any]) -> BaselineEncoder:
    """2D or 2.5D (``slice_mode``) ResNet-18; see builder.py for the contract."""
    return build_slice_mil_encoder(config, default_timm_name=DEFAULT_TIMM_NAME, display_name="ResNet-18",
                                   default_lora_targets=LORA_TARGETS)
