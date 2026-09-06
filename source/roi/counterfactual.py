from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


def _as_mask(mask: Tensor, volume: Tensor) -> Tensor:
    if mask.ndim == volume.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != volume.ndim:
        raise ValueError(f"mask rank {mask.ndim} does not match volume rank {volume.ndim}")
    if mask.shape[0] != volume.shape[0] or tuple(mask.shape[-3:]) != tuple(volume.shape[-3:]):
        raise ValueError(f"mask shape {tuple(mask.shape)} is incompatible with volume {tuple(volume.shape)}")
    if mask.shape[1] not in {1, volume.shape[1]}:
        raise ValueError("mask channel dimension must be one or match the volume")
    return (mask > 0.5).to(device=volume.device, dtype=volume.dtype)


def _masked_mean(volume: Tensor, selector: Tensor) -> Tensor:
    if selector.shape[1] == 1 and volume.shape[1] != 1:
        selector = selector.expand(-1, volume.shape[1], -1, -1, -1)
    axes = tuple(range(2, volume.ndim))
    count = selector.sum(dim=axes, keepdim=True)
    total = (volume * selector).sum(dim=axes, keepdim=True)
    fallback = volume.mean(dim=axes, keepdim=True)
    return torch.where(count > 0, total / count.clamp_min(1), fallback)


def _local_reference(volume: Tensor, mask: Tensor, operation: str, radius: int) -> tuple[Tensor, Tensor]:
    kernel = 2 * int(radius) + 1
    if operation == "remove":
        expanded = functional.max_pool3d(mask, kernel, stride=1, padding=radius)
        selector = (expanded - mask).clamp_min(0)
    else:
        # For keep-only inputs the retained anatomy is the only defensible local source.
        boundary = functional.max_pool3d(1 - mask, kernel, stride=1, padding=radius)
        selector = mask * boundary
        if not bool(selector.any()):
            selector = mask
    return _masked_mean(volume, selector), selector


@dataclass(frozen=True)
class MaskingPolicy:
    name: str = "local_mean"
    local_radius_voxels: int = 3
    noise_scale: float = 1.0

    def replacement(
        self,
        volume: Tensor,
        mask: Tensor,
        *,
        operation: str,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.name == "zero":
            return torch.zeros_like(volume)
        axes = tuple(range(2, volume.ndim))
        if self.name == "global_mean":
            mean = volume.mean(dim=axes, keepdim=True)
            return mean.expand_as(volume)
        mean, selector = _local_reference(volume, mask, operation, self.local_radius_voxels)
        if self.name == "local_mean":
            return mean.expand_as(volume)
        if self.name == "noise_matched":
            if selector.shape[1] == 1 and volume.shape[1] != 1:
                selector = selector.expand(-1, volume.shape[1], -1, -1, -1)
            variance = _masked_mean((volume - mean) ** 2, selector)
            noise = torch.randn(
                volume.shape,
                dtype=volume.dtype,
                device=volume.device,
                generator=generator,
            )
            return mean + noise * variance.clamp_min(0).sqrt() * float(self.noise_scale)
        raise ValueError(f"unsupported masking policy: {self.name}")


def apply_mask_transform(
    volume: Tensor,
    mask: Tensor,
    *,
    operation: str,
    policy: MaskingPolicy | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    binary = _as_mask(mask, volume)
    if operation not in {"keep", "remove"}:
        raise ValueError("operation must be 'keep' or 'remove'")
    replacement = (policy or MaskingPolicy()).replacement(
        volume, binary, operation=operation, generator=generator
    )
    return (
        volume * binary + replacement * (1 - binary)
        if operation == "keep"
        else volume * (1 - binary) + replacement * binary
    )

