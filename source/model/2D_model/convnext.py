"""ConvNeXt-T slice backbone for the 2D / 2.5D slice-MIL baselines."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from source.model.base import BaselineEncoder

from .builder import build_slice_mil_encoder

DEFAULT_TIMM_NAME = "convnext_tiny.fb_in22k_ft_in1k"
# timm ConvNeXt stage 4: downsample conv and the block MLP Linears (conv_dw is depthwise and
# skipped by inject_lora), as convnext_3d.
LORA_TARGETS = ("stages.3",)


def build_convnext_2d(config: Mapping[str, Any]) -> BaselineEncoder:
    """2D or 2.5D (``slice_mode``) ConvNeXt-T; see builder.py for the contract."""
    return build_slice_mil_encoder(config, default_timm_name=DEFAULT_TIMM_NAME, display_name="ConvNeXt-T",
                                   default_lora_targets=LORA_TARGETS)
