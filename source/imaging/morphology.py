"""Memory-bounded physical-space morphology for full-resolution CT masks.

``scipy.ndimage.binary_erosion``/``binary_dilation`` precompute one offset table per
border configuration: ``prod(min(shape, footprint)) * footprint_size * 8`` bytes. A 15 mm
ball at 0.7 x 0.7 x 1.0 mm spacing (45 x 45 x 31, 29 367 voxels) needs ~14.7 GB before any
work starts, and the run time is O(voxels x footprint). For a physical ball the same
result follows exactly from a Euclidean distance transform; processing z-slabs keeps the
peak memory of that transform around 1-2 GB for a 512 x 512 CTPA.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def within_distance(
    features: np.ndarray,
    spacing: Sequence[float],
    radius_mm: float,
    *,
    outside_is_feature: bool = False,
    slab_slices: int = 48,
) -> np.ndarray:
    """Return voxels whose physical distance to the nearest feature voxel is <= ``radius_mm``.

    This equals ``binary_dilation(features, structure=ball)`` for the physical ball
    ``{d : |d * spacing| <= radius_mm}`` with ``border_value=0``. With
    ``outside_is_feature=True`` every voxel beyond the array border also counts as a
    feature, so ``mask & within_distance(~mask, ..., outside_is_feature=True)`` is the
    shell removed by ``binary_erosion(mask, structure=ball, border_value=0)``.

    The volume is processed in slabs along the third axis. Each slab is extended by
    ``ceil(radius / spacing_z) + 1`` slices, which contains every feature that can lie
    within ``radius_mm`` of the slab core, so the result is exact rather than tiled.
    """
    from scipy import ndimage

    binary = np.asarray(features, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"within_distance expects a 3D mask, got shape={binary.shape}")
    sampling = tuple(float(value) for value in tuple(spacing)[:3])
    if len(sampling) != 3 or any(not math.isfinite(value) or value <= 0 for value in sampling):
        raise ValueError(f"spacing must contain three positive values, got {spacing!r}")
    radius = float(radius_mm)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(f"radius_mm must be a non-negative number, got {radius_mm!r}")
    if int(slab_slices) < 1:
        raise ValueError("slab_slices must be positive")

    result = np.zeros(binary.shape, dtype=bool)
    depth = binary.shape[2]
    margin = math.ceil(radius / sampling[2]) + 1
    for start in range(0, depth, int(slab_slices)):
        stop = min(depth, start + int(slab_slices))
        low, high = max(0, start - margin), min(depth, stop + margin)
        block = binary[:, :, low:high]
        pad_before = 0
        if outside_is_feature:
            # x/y borders are always real array borders; a z border is real only at the
            # first/last slab. Interior slab cuts must not be mistaken for the outside.
            pad_before = 1 if low == 0 else 0
            pad_after = 1 if high == depth else 0
            block = np.pad(
                block, ((1, 1), (1, 1), (pad_before, pad_after)), mode="constant", constant_values=True
            )
            offset = 1
        else:
            if not block.any():
                continue
            offset = 0
        # distance_transform_edt measures, for every True voxel, the distance to the nearest
        # False voxel; features are therefore passed as False.
        distance = ndimage.distance_transform_edt(~block, sampling=sampling)
        z_first = pad_before + (start - low)
        core = distance[
            offset : offset + binary.shape[0],
            offset : offset + binary.shape[1],
            z_first : z_first + (stop - start),
        ]
        result[:, :, start:stop] = core <= radius
        del distance, core, block
    return result


def boundary_shell(mask: np.ndarray, spacing: Sequence[float], thickness_mm: float) -> np.ndarray:
    """Voxels of ``mask`` within ``thickness_mm`` of its outside (array border counts as outside).

    Identical to ``mask & ~binary_erosion(mask, physical_ball(thickness_mm), border_value=0)``.
    """
    binary = np.asarray(mask, dtype=bool)
    return binary & within_distance(~binary, spacing, thickness_mm, outside_is_feature=True)


def physical_dilation(mask: np.ndarray, spacing: Sequence[float], radius_mm: float) -> np.ndarray:
    """Identical to ``binary_dilation(mask, physical_ball(radius_mm), border_value=0)``."""
    return within_distance(mask, spacing, radius_mm, outside_is_feature=False)


def physical_ball(spacing: Sequence[float], radius_mm: float) -> np.ndarray:
    """Structuring element of all grid offsets within ``radius_mm``; reference/testing only.

    Do not pass a large ball to scipy binary morphology on a full CT; use the functions
    above instead.
    """
    voxel = np.asarray(tuple(spacing)[:3], dtype=float)
    radii = np.maximum(1, np.ceil(float(radius_mm) / voxel).astype(int))
    grids = np.ogrid[tuple(slice(-int(value), int(value) + 1) for value in radii)]
    distance = sum((grid * voxel[index]) ** 2 for index, grid in enumerate(grids))
    return np.asarray(distance <= float(radius_mm) ** 2, dtype=bool)


__all__ = ["boundary_shell", "physical_ball", "physical_dilation", "within_distance"]
