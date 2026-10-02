"""Swin-T slice backbone for the 2D / 2.5D slice-MIL baselines."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from source.model.base import BaselineEncoder

from .builder import build_slice_mil_encoder

DEFAULT_TIMM_NAME = "swin_tiny_patch4_window7_224.ms_in1k"
LORA_TARGETS = ("attn.qkv", "attn.proj")


def build_swin_2d(config: Mapping[str, Any]) -> BaselineEncoder:
    """2D or 2.5D (``slice_mode``) Swin-T; see builder.py for the contract."""
    return build_slice_mil_encoder(config, default_timm_name=DEFAULT_TIMM_NAME, display_name="Swin-T",
                                   default_lora_targets=LORA_TARGETS)
