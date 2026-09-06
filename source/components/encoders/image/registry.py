from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from .base import BaseImageEncoder


Builder = Callable[[Mapping[str, Any]], BaseImageEncoder]
_REGISTRY: dict[str, Builder] = {}


def register_backbone(name: str, builder: Builder) -> None:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized or normalized in _REGISTRY:
        raise ValueError(f"backbone already registered or invalid: {name}")
    _REGISTRY[normalized] = builder


def _ct_fm(config: Mapping[str, Any]) -> BaseImageEncoder:
    from .ct_fm import build_ct_fm

    return build_ct_fm(config)


def _ct_clip(config: Mapping[str, Any]) -> BaseImageEncoder:
    from .ct_clip import build_ct_clip

    return build_ct_clip(config)


def _totalfm(config: Mapping[str, Any]) -> BaseImageEncoder:
    from .totalfm import build_totalfm

    return build_totalfm(config)


register_backbone("ct_fm", _ct_fm)
register_backbone("ct_clip", _ct_clip)
register_backbone("totalfm", _totalfm)


def registered_backbones() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_image_encoder(config: Mapping[str, Any]) -> BaseImageEncoder:
    name = str(config.get("backbone") or "").strip().lower().replace("-", "_")
    if name not in _REGISTRY:
        raise KeyError(f"unknown image backbone {name!r}; available={registered_backbones()}")
    return _REGISTRY[name](config)
