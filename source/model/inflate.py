"""2-D -> 3-D weight inflation (I3D style) for ImageNet-initialised volumetric baselines.

Every 3-D model here sees the cache's RAS volume ``[B, C, x, y, z]`` (``source/model/base.py``),
so its kernels are ``[O, I, kx, ky, kz]`` and z (inferior -> superior) is the through-plane
axis of the axial slices. A 2-D ImageNet kernel ``[O, I, kh, kw]`` therefore lies in the axial
(x, y) plane (kh -> x, kw -> y, the same orientation the 2-D slice-MIL arms feed an axial
slice ``volume[..., z]`` to timm) and is repeated ``kz`` times along z and divided by ``kz``,
so a volume that is constant along z produces the 2-D network's response on each axial
slice. Position tables are placed the same way. RGB stems are collapsed to one channel by
summing over the input channels, which is exactly the 2-D network applied to a grey image
replicated into three channels.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def collapse_input_channels(weight: Tensor, channels: int) -> Tensor:
    """[O, I, ...] -> [O, channels, ...] by summing (channels=1) or tiling the mean."""
    if weight.shape[1] == channels:
        return weight
    if channels == 1:
        return weight.sum(dim=1, keepdim=True)
    mean = weight.mean(dim=1, keepdim=True)
    return mean.repeat(1, channels, *([1] * (weight.ndim - 2))) * (weight.shape[1] / channels)


def inflate_kernel(weight2d: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    """Inflate a 4-D conv kernel ``[O, I, kh, kw]`` to ``target_shape`` ([O, I, kx, ky, kz]).

    (kh, kw) is placed on the axial (x, y) plane and repeated along z (the last axis).
    """
    if weight2d.ndim != 4 or len(target_shape) != 5:
        raise ValueError(f"cannot inflate {tuple(weight2d.shape)} to {target_shape}")
    out_channels, in_channels, size_x, size_y, depth = target_shape
    weight = collapse_input_channels(weight2d, in_channels) if weight2d.shape[1] != in_channels else weight2d
    if weight.shape[0] != out_channels:
        raise ValueError(f"output channels differ: {tuple(weight2d.shape)} vs {target_shape}")
    if tuple(weight.shape[-2:]) != (size_x, size_y):
        weight = F.interpolate(weight.float(), size=(size_x, size_y), mode="bilinear", align_corners=False)
    return weight.unsqueeze(-1).repeat(1, 1, 1, 1, depth) / depth


def inflate_state_dict(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    *,
    rename: Callable[[str], str | None] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Map a 2-D state dict onto a 3-D model's state dict.

    ``rename`` maps a source key to the target key (None drops it). A tensor is copied when
    shapes agree, inflated when a 4-D kernel meets a 5-D one, and skipped (reported) otherwise.
    """
    converted: dict[str, Tensor] = {}
    copied: list[str] = []
    inflated: list[str] = []
    skipped: list[str] = []
    for key, value in source.items():
        name = rename(key) if rename is not None else key
        if name is None:
            continue
        if name not in target:
            skipped.append(f"{key}: no target")
            continue
        wanted = tuple(target[name].shape)
        if tuple(value.shape) == wanted:
            converted[name] = value
            copied.append(name)
        elif value.ndim == 4 and len(wanted) == 5:
            try:
                converted[name] = inflate_kernel(value, wanted)
                inflated.append(name)
            except ValueError as exc:
                skipped.append(f"{key}: {exc}")
        else:
            skipped.append(f"{key}: {tuple(value.shape)} vs {wanted}")
    return converted, {"copied": len(copied), "inflated": len(inflated), "skipped": skipped}


def interpolate_positions_2d_to_3d(pos2d: Tensor, grid2d: tuple[int, int], grid3d: tuple[int, int, int]) -> Tensor:
    """[1, H*W, C] 2-D position table -> [1, X*Y*Z, C] for a token grid ``grid3d = (X, Y, Z)``.

    The table is resized to the axial (X, Y) plane and repeated along Z; tokens are ordered
    as ``Conv3d(...)(volume).flatten(2)`` orders them (x slowest, z fastest).
    """
    channels = pos2d.shape[-1]
    table = pos2d.reshape(1, grid2d[0], grid2d[1], channels).permute(0, 3, 1, 2).float()
    table = F.interpolate(table, size=tuple(grid3d[:2]), mode="bicubic", align_corners=False)
    table = table.unsqueeze(-1).repeat(1, 1, 1, 1, grid3d[2])  # [1, C, X, Y, Z]
    return table.flatten(2).transpose(1, 2).to(pos2d.dtype)


def interpolate_positions_3d(pos: Tensor, source: tuple[int, int, int], target: tuple[int, int, int]) -> Tensor:
    """[1, D*H*W, C] 3-D position table resized trilinearly to another grid."""
    if tuple(source) == tuple(target):
        return pos
    channels = pos.shape[-1]
    table = pos.reshape(1, *source, channels).permute(0, 4, 1, 2, 3).float()
    table = F.interpolate(table, size=target, mode="trilinear", align_corners=False)
    return table.flatten(2).transpose(1, 2).to(pos.dtype)


def timm_create_with_report(name: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """``timm.create_model(name, pretrained=True, ...)`` plus what its internal load did.

    timm loads ImageNet weights non-strictly when the classifier is dropped, and does not
    return the result. The top-level ``load_state_dict`` call is observed instead, so the
    reported missing / unexpected keys are the real ones.
    """
    import timm
    from torch import nn

    calls: list[tuple[nn.Module, int, Any]] = []
    original = nn.Module.load_state_dict

    def observe(module, state_dict, *args, **kw):
        result = original(module, state_dict, *args, **kw)
        calls.append((module, len(state_dict), result))
        return result

    nn.Module.load_state_dict = observe  # type: ignore[method-assign]
    try:
        model = timm.create_model(name, pretrained=True, **kwargs)
    finally:
        nn.Module.load_state_dict = original  # type: ignore[method-assign]
    top = next((call for call in calls if call[0] is model), calls[-1] if calls else None)
    if top is None:
        raise RuntimeError(f"timm reported no weight load for {name}")
    _, checkpoint_tensors, result = top
    cfg = dict(getattr(model, "pretrained_cfg", {}) or {})
    return model, {
        "source": f"timm:{name} ({cfg.get('hf_hub_id') or cfg.get('url') or name})",
        "checkpoint_tensors": checkpoint_tensors,
        "missing_keys": list(getattr(result, "missing_keys", []) or []),
        "unexpected_keys": list(getattr(result, "unexpected_keys", []) or []),
    }


@torch.no_grad()
def timm_state_dict(name: str) -> tuple[dict[str, Tensor], str]:
    """ImageNet weights of a timm model (downloaded to the Hugging Face cache on first use)."""
    import timm

    model = timm.create_model(name, pretrained=True, num_classes=0)
    tag = getattr(getattr(model, "pretrained_cfg", None), "get", lambda *_: None)("hf_hub_id") or name
    return {key: value.detach().clone() for key, value in model.state_dict().items()}, f"timm:{tag}"


__all__ = [
    "collapse_input_channels",
    "inflate_kernel",
    "inflate_state_dict",
    "interpolate_positions_2d_to_3d",
    "interpolate_positions_3d",
    "timm_state_dict",
]
