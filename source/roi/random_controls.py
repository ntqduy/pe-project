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


def _nearest_volume(
    candidate: np.ndarray,
    target_voxels: int,
    spacing: Sequence[float],
    seed: int,
) -> np.ndarray:
    coordinates = np.argwhere(candidate)
    if target_voxels <= 0:
        raise ValueError("target ROI is empty")
    if target_voxels > len(coordinates):
        raise ValueError(
            f"target_voxels={target_voxels} exceeds eligible_control_voxels={len(coordinates)}"
        )
    rng = np.random.default_rng(int(seed))
    anchor = coordinates[int(rng.integers(0, len(coordinates)))]
    scaled = (coordinates - anchor) * np.asarray(tuple(spacing), dtype=float)
    distances = np.einsum("ij,ij->i", scaled, scaled)
    selected = np.argpartition(distances, target_voxels - 1)[:target_voxels]
    output = np.zeros(candidate.shape, dtype=bool)
    chosen = coordinates[selected]
    output[tuple(chosen.T)] = True
    return output


def matched_random_control(
    *,
    target: np.ndarray,
    body: np.ndarray,
    body_wall: np.ndarray | None,
    spacing: Sequence[float],
    seed: int,
    exclusion_margin_mm: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an exact-volume negative control without overlapping the target anatomy.

    Body wall is preferred. If it is too small, the eligible non-target body is used and
    that fallback is recorded. No fallback is allowed to overlap the target.
    """

    target_binary = np.asarray(target, dtype=bool)
    body_binary = np.asarray(body, dtype=bool)
    if target_binary.shape != body_binary.shape:
        raise ValueError("target and body masks must share geometry")
    excluded = (
        dilate_mask(target_binary, spacing, float(exclusion_margin_mm))
        if exclusion_margin_mm > 0
        else target_binary
    )
    target_voxels = int(target_binary.sum())
    candidates: list[tuple[str, np.ndarray]] = []
    if body_wall is not None:
        wall = np.asarray(body_wall, dtype=bool)
        if wall.shape != target_binary.shape:
            raise ValueError("body-wall mask must share target geometry")
        candidates.append(("body_wall_outside_target", wall & body_binary & ~excluded))
    candidates.append(("body_outside_target", body_binary & ~excluded))
    selected_name = ""
    selected_candidate: np.ndarray | None = None
    for name, candidate in candidates:
        if int(candidate.sum()) >= target_voxels:
            selected_name = name
            selected_candidate = candidate
            break
    if selected_candidate is None:
        largest = max((int(candidate.sum()) for _, candidate in candidates), default=0)
        raise ValueError(
            f"no non-overlapping candidate can match target_voxels={target_voxels}; "
            f"largest_eligible={largest}"
        )
    control = _nearest_volume(selected_candidate, target_voxels, spacing, seed)
    overlap = int(np.logical_and(control, target_binary).sum())
    if overlap:
        raise RuntimeError(f"matched random control overlaps target by {overlap} voxels")
    return control, {
        "method": "random_anchor_nearest_physical_eligible_voxels",
        "candidate_region": selected_name,
        "fallback_from_body_wall": bool(body_wall is not None and selected_name != "body_wall_outside_target"),
        "target_voxels": target_voxels,
        "actual_voxels": int(control.sum()),
        "eligible_voxels": int(selected_candidate.sum()),
        "overlap_voxels": overlap,
        "exclusion_margin_mm": float(exclusion_margin_mm),
        "shape_matched": False,
        "volume_matched": True,
    }

