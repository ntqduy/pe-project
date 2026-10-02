"""``model.backbone`` names of the baseline zoo -> lazy builders.

Registered into ``source/components/encoders/image/registry.py`` so every existing code path
(factory, preflight, evaluate, Grad-CAM preview) resolves them like CT-FM. Builders import
their module only when called: a missing optional dependency (mamba_ssm, monai, timm)
affects only the arm that needs it.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

# name -> (module under source.model, builder function)
BASELINE_BACKBONES: dict[str, tuple[str, str]] = {
    # 2D / 2.5D slice-MIL (the registry entry chooses slice_mode)
    "resnet18_2d": ("2D_model.resnet", "build_resnet18_2d"),
    "resnet18_25d": ("2D_model.resnet", "build_resnet18_2d"),
    "convnext_2d": ("2D_model.convnext", "build_convnext_2d"),
    "convnext_25d": ("2D_model.convnext", "build_convnext_2d"),
    "vit_2d": ("2D_model.vit", "build_vit_2d"),
    "vit_25d": ("2D_model.vit", "build_vit_2d"),
    "swin_2d": ("2D_model.swin", "build_swin_2d"),
    "swin_25d": ("2D_model.swin", "build_swin_2d"),
    # 3D
    "resnet18_3d": ("3D_model.resnet", "build_resnet18_3d"),
    "resnet50_3d": ("3D_model.resnet", "build_resnet50_3d"),
    "densenet121_3d": ("3D_model.densenet", "build_densenet121_3d"),
    "convnext_3d": ("3D_model.convnext", "build_convnext_3d"),
    "vit_3d": ("3D_model.vit", "build_vit_3d"),
    "swin_3d": ("3D_model.swin", "build_swin_3d"),
    "nnmamba_3d": ("3D_model.nnmamba", "build_nnmamba_3d"),
    "mamba_mae_3d": ("3D_model.mamba_mae", "build_mamba_mae_3d"),
    "vmamba_3d": ("3D_model.vmamba", "build_vmamba_3d"),
    "penet_3d": ("3D_model.penet", "build_penet_3d"),
}


def _lazy(module: str, function: str) -> Callable[[Mapping[str, Any]], Any]:
    def build(config: Mapping[str, Any]) -> Any:
        return getattr(importlib.import_module(f"source.model.{module}"), function)(config)

    build.__name__ = function
    return build


def baseline_builders() -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    return {name: _lazy(module, function) for name, (module, function) in BASELINE_BACKBONES.items()}


__all__ = ["BASELINE_BACKBONES", "baseline_builders"]
