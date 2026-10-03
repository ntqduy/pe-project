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
    # Entry point of the baseline zoo's CT-FM arms (source/model/3D_model/ctfm.py).
    import importlib

    return importlib.import_module("source.model.3D_model.ctfm").build_ctfm_lora_3d(config)


def _ct_fm_features(config: Mapping[str, Any]) -> BaseImageEncoder:
    import importlib

    return importlib.import_module("source.model.3D_model.ctfm").build_ctfm_frozen_3d(config)


register_backbone("ct_fm", _ct_fm)
register_backbone("ct_fm_features", _ct_fm_features)


def _register_baseline_zoo() -> None:
    # 2D / 2.5D / 3D baselines (source/model); builders import their module lazily.
    from source.model.registry import baseline_builders

    for name, builder in baseline_builders().items():
        register_backbone(name, builder)


_register_baseline_zoo()


def registered_backbones() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_image_encoder(config: Mapping[str, Any]) -> BaseImageEncoder:
    name = str(config.get("backbone") or "").strip().lower().replace("-", "_")
    if name not in _REGISTRY:
        raise KeyError(f"unknown image backbone {name!r}; available={registered_backbones()}")
    return _REGISTRY[name](config)
