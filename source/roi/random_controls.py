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
    valid_ct: np.ndarray | None = None,
    spacing: Sequence[float],
    seed: int,
    exclusion_margin_mm: float = 0.0,
    max_attempts: int = 500,
    require_exact_voxel_match: bool = True,
    preserve_z_range: bool = True,
    superior_axis: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rigidly translate a source ROI to a reproducible eligible body location.

    Eligible voxels lie inside the body or the body wall, inside valid CT (``valid_ct``,
    e.g. finite HU), and outside the source ROI and the forbidden union after the physical
    exclusion margin. ``superior_axis`` is the array axis of the inferior-superior direction
    (derive it from the affine, see ``source.imaging.nifti.superior_axis``); with
    ``preserve_z_range`` the control keeps the source's extent along that axis.

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
    if valid_ct is not None and np.asarray(valid_ct).shape != target_binary.shape:
        raise ValueError("valid-CT mask must share target geometry")
    if int(superior_axis) not in (0, 1, 2):
        raise ValueError("superior_axis must be 0, 1 or 2")
    superior_axis = int(superior_axis)

    # "Inside body/body-wall": the wall shell is eligible even where the body mask in use
    # (for example the HU-threshold fallback) does not cover it.
    region = (body_binary | np.asarray(body_wall, dtype=bool)) if body_wall is not None else body_binary
    if valid_ct is not None:
        region = region & np.asarray(valid_ct, dtype=bool)
    excluded_source = target_binary | forbidden_binary
    excluded = (
        dilate_mask(excluded_source, spacing, float(exclusion_margin_mm))
        if exclusion_margin_mm > 0
        else excluded_source
    )
    candidate = region & ~excluded
    if not candidate.any():
        raise ValueError("no_valid_control_location")

    coordinates = np.argwhere(target_binary)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    # Only translations that keep the ROI's bounding box inside the candidate's bounding box
    # can succeed, so the random draws are restricted to them. This drops no valid location
    # and stops most of the max_attempts budget being spent outside the body.
    ranges = []
    for axis in range(3):
        occupied = np.flatnonzero(candidate.any(axis=tuple(other for other in range(3) if other != axis)))
        ranges.append(
            np.arange(
                int(occupied[0]) - int(low[axis]),
                int(occupied[-1]) - int(high[axis]) + 1,
                dtype=int,
            )
        )
    if preserve_z_range:
        ranges[superior_axis] = (
            np.asarray([0], dtype=int)
            if np.any(ranges[superior_axis] == 0)
            else np.asarray([], dtype=int)
        )
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

    # A translated ROI is valid iff every shifted voxel lies in ``candidate`` (= inside the
    # body and outside the dilated exclusion), which is exactly the old three whole-volume
    # checks. Testing only the ROI's own voxels avoids building a 512x512xZ array per
    # attempt; a small spread subset is tried first so most rejections are cheap.
    tested = 0
    chosen_offset: tuple[int, int, int] | None = None
    probe = coordinates[:: max(1, len(coordinates) // 2048)]
    for offset in sampled:
        tested += 1
        shift = np.asarray(offset, dtype=int)
        if not candidate[tuple((probe + shift).T)].all():
            continue
        if not candidate[tuple((coordinates + shift).T)].all():
            continue
        chosen_offset = offset
        break
    if chosen_offset is None:
        raise ValueError("no_valid_control_location")
    control = _translated(target_binary, chosen_offset)

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
        "candidate_region": (
            ("body_or_body_wall" if body_wall is not None else "body")
            + ("_and_valid_ct" if valid_ct is not None else "")
            + "_outside_forbidden_union"
        ),
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
        "superior_axis": superior_axis,
        "shape_matched": True,
        "volume_matched": actual_voxels == target_voxels,
        "target_physical_volume_mm3": target_volume_mm3,
        "control_physical_volume_mm3": control_volume_mm3,
        "physical_volume_error_mm3": control_volume_mm3 - target_volume_mm3,
        "require_exact_voxel_match": bool(require_exact_voxel_match),
    }
