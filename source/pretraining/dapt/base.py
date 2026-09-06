from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from source.components.encoders.image.base import BaseImageEncoder


class DAPTObjective(nn.Module):
    def __init__(self, encoder: BaseImageEncoder):
        super().__init__()
        self.image_encoder = encoder


def build_dapt(name: str, encoder: BaseImageEncoder, config: Mapping[str, Any] | None = None) -> DAPTObjective:
    normalized = name.strip().lower()
    options = dict(config or {})
    if normalized in {"none", "d00"}:
        from .none import NoDAPT

        return NoDAPT(encoder)
    if normalized in {"mae", "d01"}:
        from .mae import MaskedAutoencoderDAPT

        return MaskedAutoencoderDAPT(encoder, mask_ratio=float(options.get("mask_ratio", 0.75)))
    if normalized in {"dino", "d02"}:
        from .dino import DINODAPT

        return DINODAPT(encoder, projection_dim=int(options.get("projection_dim", 128)))
    if normalized in {"simclr", "d03"}:
        from .simclr import SimCLRDAPT

        return SimCLRDAPT(encoder, projection_dim=int(options.get("projection_dim", 128)), temperature=float(options.get("temperature", 0.1)))
    if normalized in {"anatomy", "anatomy_dapt", "d04"}:
        from .anatomy_dapt import AnatomyAwareDAPT

        return AnatomyAwareDAPT(encoder, projection_dim=int(options.get("projection_dim", 128)))
    raise ValueError(f"unsupported DAPT method: {name}")
