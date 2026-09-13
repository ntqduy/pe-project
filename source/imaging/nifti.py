from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def combine_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(mask).astype(bool) for mask in masks]
    if not arrays:
        raise ValueError("at least one mask is required")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("masks must share a shape")
    return np.logical_or.reduce(arrays).astype(np.uint8)


def load_nifti(path: str | Path) -> tuple[np.ndarray, Any]:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel is required for NIfTI mask processing") from exc
    image = nib.load(str(path))
    return np.asarray(image.dataobj), image


def save_binary_mask(mask: np.ndarray, reference: Any, destination: str | Path) -> Path:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel is required for NIfTI mask processing") from exc
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(binary, reference.affine, header), str(path))
    return path


def geometry_metadata(image: Any) -> dict[str, Any]:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel is required for NIfTI geometry inspection") from exc
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return {
        "shape": [int(value) for value in image.shape[:3]],
        "spacing": list(spacing),
        "orientation": list(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine, dtype=float).round(8).tolist(),
    }


def same_geometry(first: Any, second: Any, *, affine_tolerance: float = 1e-5) -> bool:
    return tuple(first.shape[:3]) == tuple(second.shape[:3]) and np.allclose(
        first.affine, second.affine, atol=affine_tolerance, rtol=0
    )


def binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("Dice masks must have equal shape")
    denominator = int(left.sum() + right.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(left, right).sum() / denominator)


def mask_qc(
    mask_path: str | Path,
    volume_path: str | Path,
    *,
    minimum_volume_ml: float | None = None,
    maximum_volume_ml: float | None = None,
) -> dict[str, Any]:
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise RuntimeError("SciPy is required for mask connected-component QC") from exc
    try:
        mask, mask_image = load_nifti(mask_path)
        _, volume_image = load_nifti(volume_path)
    except Exception as exc:  # noqa: BLE001 - unreadable/corrupt images are QC FAIL results
        return {"status": "FAIL", "reason": f"unreadable: {type(exc).__name__}: {exc}"}
    unique = set(np.unique(mask).tolist())
    geometry = geometry_metadata(mask_image)
    if not same_geometry(mask_image, volume_image):
        return {**geometry, "status": "FAIL", "reason": "geometry_mismatch"}
    if not unique <= {0, 1}:
        return {**geometry, "status": "FAIL", "reason": f"non_binary_values={sorted(unique)}"}
    binary = mask.astype(bool)
    voxels = int(binary.sum())
    spacing = tuple(float(value) for value in mask_image.header.get_zooms()[:3])
    volume_ml = float(voxels * np.prod(spacing) / 1000.0)
    components = int(ndimage.label(binary)[1]) if voxels else 0
    status = "PASS"
    reasons: list[str] = []
    if voxels == 0:
        status = "FAIL"
        reasons.append("empty")
    if minimum_volume_ml is not None and volume_ml < minimum_volume_ml:
        status = "SUSPICIOUS" if status == "PASS" else status
        reasons.append(f"volume_below_{minimum_volume_ml:g}ml")
    if maximum_volume_ml is not None and volume_ml > maximum_volume_ml:
        status = "SUSPICIOUS" if status == "PASS" else status
        reasons.append(f"volume_above_{maximum_volume_ml:g}ml")
    return {
        **geometry,
        "status": status,
        "reason": ";".join(reasons) or "basic_qc_pass",
        "voxel_count": voxels,
        "volume_ml": volume_ml,
        "component_count": components,
    }
