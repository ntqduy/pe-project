from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy import ndimage


def require_same_shape(masks: Iterable[np.ndarray]) -> list[np.ndarray]:
    arrays = [np.asarray(mask, dtype=bool) for mask in masks]
    if not arrays:
        raise ValueError("at least one mask is required")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all masks must share the same geometry")
    return arrays


def union_masks(*masks: np.ndarray) -> np.ndarray:
    arrays = require_same_shape(masks)
    return np.logical_or.reduce(arrays)


def subtract_masks(base: np.ndarray, *remove: np.ndarray) -> np.ndarray:
    arrays = require_same_shape((base, *remove))
    excluded = np.logical_or.reduce(arrays[1:]) if len(arrays) > 1 else np.zeros_like(arrays[0])
    return arrays[0] & ~excluded


def physical_ball(spacing: Sequence[float], radius_mm: float) -> np.ndarray:
    spacing_array = np.asarray(tuple(spacing), dtype=float)
    if spacing_array.shape != (3,) or not np.isfinite(spacing_array).all() or (spacing_array <= 0).any():
        raise ValueError("spacing must contain three positive finite values")
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("radius_mm must be positive and finite")
    radii = np.maximum(1, np.ceil(float(radius_mm) / spacing_array).astype(int))
    grids = np.ogrid[tuple(slice(-int(value), int(value) + 1) for value in radii)]
    squared = sum(
        (grid * spacing_array[index] / float(radius_mm)) ** 2
        for index, grid in enumerate(grids)
    )
    return np.asarray(squared <= 1.0, dtype=bool)


def dilate_mask(mask: np.ndarray, spacing: Sequence[float], margin_mm: float) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if margin_mm == 0:
        return binary.copy()
    return ndimage.binary_dilation(binary, structure=physical_ball(spacing, margin_mm))


def body_mask_from_hu(volume: np.ndarray, threshold_hu: float = -900.0) -> np.ndarray:
    candidate = np.isfinite(volume) & (np.asarray(volume) > float(threshold_hu))
    labels, count = ndimage.label(candidate)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        candidate = labels == int(np.argmax(sizes))
    return ndimage.binary_fill_holes(candidate)

