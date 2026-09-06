from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def resize_mask(mask: Tensor, spatial_shape: tuple[int, int, int]) -> Tensor:
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError(f"mask must be [B,1,D,H,W], got {tuple(mask.shape)}")
    return functional.interpolate(mask.float(), size=spatial_shape, mode="nearest").clamp_(0, 1)


def mask_guided_pool(feature_map: Tensor, mask: Tensor, epsilon: float = 1e-6) -> tuple[Tensor, Tensor]:
    if feature_map.ndim != 5:
        raise ValueError(f"feature map must be [B,C,D,H,W], got {tuple(feature_map.shape)}")
    resized = resize_mask(mask, tuple(feature_map.shape[-3:])).to(feature_map.device, feature_map.dtype)
    denominator = resized.sum(dim=(2, 3, 4))
    pooled = (feature_map * resized).sum(dim=(2, 3, 4)) / denominator.clamp_min(epsilon)
    present = denominator.squeeze(1) > epsilon
    pooled = torch.where(present[:, None], pooled, torch.zeros_like(pooled))
    return pooled, present
