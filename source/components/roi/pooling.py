from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor


def resize_mask(mask: Tensor, spatial_shape: tuple[int, int, int]) -> Tensor:
    """Region mask -> per-cell weights in [0, 1] on a feature grid.

    Axes that shrink (the usual case: a 128^3 input against a 4^3..16^3 feature grid) are
    averaged over each cell (``adaptive_avg_pool3d``), so a weight is the fraction of that
    cell inside the region. Nearest-neighbour sampling would read a single corner voxel per
    cell (0, 16, 32, ... for 128 -> 8) and drop or displace small regions such as the PA.
    Axes that grow are filled by nearest-neighbour, which keeps a binary mask binary.
    """
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError(f"mask must be [B,1,D,H,W], got {tuple(mask.shape)}")
    values = mask.float().clamp(0, 1)
    source = tuple(values.shape[-3:])
    target = tuple(int(size) for size in spatial_shape)
    pooled_shape = tuple(min(have, want) for have, want in zip(source, target))
    if pooled_shape != source:
        values = functional.adaptive_avg_pool3d(values, pooled_shape)
    if pooled_shape != target:
        values = functional.interpolate(values, size=target, mode="nearest")
    return values.clamp(0, 1)


def select_mask_slices(mask: Tensor, slice_groups: Sequence[Sequence[int]], slice_depth: int | None = None) -> Tensor:
    """[B,1,x,y,z] -> [B,1,x,y,N] for a slice-MIL feature map [B,C,h,w,N].

    Instance ``i`` was encoded from the axial slices ``slice_groups[i]`` (one slice in 2D,
    the adjacent triplet in 2.5D; ``source/model/2D_model/mil.py``), so its weight plane is
    the mask averaged over exactly those slices, not over an arbitrary depth window.
    ``slice_depth`` is the z size the indices refer to (the input volume's).
    """
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.ndim != 5:
        raise ValueError(f"mask must be [B,1,x,y,z], got {tuple(mask.shape)}")
    values = mask.float()
    if slice_depth is not None and values.shape[-1] != int(slice_depth):
        values = resize_mask(values, (values.shape[-3], values.shape[-2], int(slice_depth)))
    index = torch.as_tensor([list(group) for group in slice_groups], dtype=torch.long, device=values.device)
    return values[..., index].mean(dim=-1)                                # [B, 1, x, y, N]


def mask_guided_pool(
    feature_map: Tensor,
    mask: Tensor,
    epsilon: float = 1e-6,
    *,
    slice_groups: Sequence[Sequence[int]] | None = None,
    slice_depth: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Mask-weighted mean ``sum(f * w) / sum(w)`` of a feature map, with soft weights w.

    A region is present when any voxel of its (slice-selected) input mask exceeds ``epsilon``.
    Presence is read on the mask itself, not on the cell-averaged weights: one voxel gives
    1/cell_volume of a cell, which a fixed threshold on ``sum(w)`` would drop once a cell holds
    more than 1/epsilon voxels (512x512x300 -> 4^3 is ~1.2e6 voxels per cell).
    """
    if feature_map.ndim != 5:
        raise ValueError(f"feature map must be [B,C,D,H,W], got {tuple(feature_map.shape)}")
    if slice_groups is not None:
        mask = select_mask_slices(mask, slice_groups, slice_depth)
        if mask.shape[-1] != feature_map.shape[-1]:
            raise ValueError(
                f"{mask.shape[-1]} slice groups for a feature map with {feature_map.shape[-1]} slices"
            )
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    present = (mask.float().clamp(0, 1).flatten(1).amax(dim=1) > epsilon).to(feature_map.device)
    weights = resize_mask(mask, tuple(feature_map.shape[-3:])).to(feature_map.device)
    denominator = weights.sum(dim=(2, 3, 4))
    present = present & (denominator.squeeze(1) > 0)
    safe = torch.where(present[:, None], denominator, torch.ones_like(denominator))
    pooled = (feature_map.float() * weights).sum(dim=(2, 3, 4)) / safe
    pooled = torch.where(present[:, None], pooled, torch.zeros_like(pooled))
    return pooled.to(feature_map.dtype), present
