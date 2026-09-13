from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from .masks import dilate_mask


def stable_control_seed(seed: int, patient_id: str, study_id: str, control_for: str) -> int:
    identity = f"{int(seed)}|{patient_id}|{study_id}|{control_for}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _translated(mask: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    coordinates = np.argwhere(mask)
    shifted = coordinates + np.asarray(offset, dtype=int)
    output = np.zeros(mask.shape, dtype=bool)
    output[tuple(shifted.T)] = True
    return output


def matched_random_control(
    *,
    target: np.ndarray,
    body: np.ndarray,
    body_wall: np.ndarray | None,
    forbidden: np.ndarray | None = None,
    spacing: Sequence[float],
    seed: int,
    exclusion_margin_mm: float = 0.0,
    max_attempts: int = 500,
    require_exact_voxel_match: bool = True,
    preserve_z_range: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rigidly translate a source ROI to a reproducible eligible body location.

    The source shape is never eroded, cropped, resampled, or filled with arbitrary nearest
    voxels. If no valid translation exists, callers must record a failed ROI rather than an
    empty successful mask.
    """

    target_binary = np.asarray(target, dtype=bool)
    body_binary = np.asarray(body, dtype=bool)
    if target_binary.ndim != 3 or target_binary.shape != body_binary.shape:
        raise ValueError("target and body masks must share 3-D geometry")
    target_voxels = int(target_binary.sum())
    if target_voxels <= 0:
        raise ValueError("target_roi_empty")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if len(tuple(spacing)) != 3 or any(float(value) <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values")

    forbidden_binary = (
        np.zeros(target_binary.shape, dtype=bool)
        if forbidden is None
        else np.asarray(forbidden, dtype=bool)
    )
    if forbidden_binary.shape != target_binary.shape:
        raise ValueError("forbidden anatomy and target masks must share geometry")
    if body_wall is not None and np.asarray(body_wall).shape != target_binary.shape:
        raise ValueError("body-wall mask must share target geometry")

    excluded_source = target_binary | forbidden_binary
    excluded = (
        dilate_mask(excluded_source, spacing, float(exclusion_margin_mm))
        if exclusion_margin_mm > 0
        else excluded_source
    )
    candidate = body_binary & ~excluded

    coordinates = np.argwhere(target_binary)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    ranges = [
        np.arange(-int(low[axis]), int(target_binary.shape[axis] - high[axis]), dtype=int)
        for axis in range(3)
    ]
    if preserve_z_range:
        ranges[2] = np.asarray([0], dtype=int)
    rng = np.random.default_rng(int(seed))
    shape = tuple(len(values) for values in ranges)
    total_offsets = int(np.prod(shape))
    zero_is_possible = all(np.any(values == 0) for values in ranges)
    available_offsets = total_offsets - int(zero_is_possible)
    if available_offsets <= 0:
        raise ValueError("no_valid_control_location")

    # Do not materialize an X*Y*Z mesh: clinical CTPA grids can make that hundreds of
    # millions of offsets. Sample unique flat indices and decode only the candidates
    # that will actually be tested.
    requested = min(int(max_attempts), available_offsets)
    sampled: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    while len(sampled) < requested:
        flat = int(rng.integers(0, total_offsets))
        if flat in seen:
            continue
        seen.add(flat)
        position = np.unravel_index(flat, shape)
        offset = tuple(int(ranges[axis][position[axis]]) for axis in range(3))
        if offset == (0, 0, 0):
            continue
        sampled.append(offset)

    tested = 0
    control: np.ndarray | None = None
    chosen_offset: tuple[int, int, int] | None = None
    for offset in sampled:
        tested += 1
        proposal = _translated(target_binary, offset)
        if np.any(proposal & ~body_binary) or np.any(proposal & excluded):
            continue
        if not np.all(proposal <= candidate):
            continue
        control = proposal
        chosen_offset = offset
        break
    if control is None or chosen_offset is None:
        raise ValueError("no_valid_control_location")

    actual_voxels = int(control.sum())
    if require_exact_voxel_match and actual_voxels != target_voxels:
        raise RuntimeError("control_voxel_count_mismatch")
    source_overlap = int(np.logical_and(control, target_binary).sum())
    forbidden_overlap = int(np.logical_and(control, forbidden_binary).sum())
    dice = 2.0 * source_overlap / (actual_voxels + target_voxels)
    voxel_volume_mm3 = float(np.prod(tuple(float(value) for value in spacing)))
    target_volume_mm3 = target_voxels * voxel_volume_mm3
    control_volume_mm3 = actual_voxels * voxel_volume_mm3
    return control, {
        "method": "seeded_rigid_translation",
        "candidate_region": "body_and_valid_ct_outside_forbidden_union",
        "target_voxels": target_voxels,
        "actual_voxels": actual_voxels,
        "eligible_voxels": int(candidate.sum()),
        "overlap_voxels": source_overlap,
        "forbidden_overlap_voxels": forbidden_overlap,
        "intersection_voxels": source_overlap,
        "dice_with_source": dice,
        "exclusion_margin_mm": float(exclusion_margin_mm),
        "translation_voxels": list(chosen_offset),
        "attempts_tested": tested,
        "max_attempts": int(max_attempts),
        "preserve_z_range": bool(preserve_z_range),
        "shape_matched": True,
        "volume_matched": actual_voxels == target_voxels,
        "target_physical_volume_mm3": target_volume_mm3,
        "control_physical_volume_mm3": control_volume_mm3,
        "physical_volume_error_mm3": control_volume_mm3 - target_volume_mm3,
        "require_exact_voxel_match": bool(require_exact_voxel_match),
    }
