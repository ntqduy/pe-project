"""Put native-geometry masks on the grid of a preprocessed model input.

Segmentation and ROI masks are written in the geometry of the original CT (e.g. LAS,
512 x 512 x Z). Model inputs are preprocessed caches: reoriented, resampled, body-cropped
and fitted/padded (``volumes/*.npy``), or CT-FM feature grids (``ct_fm/features/*.npy``).
Their sidecar ``<file>.metadata.json`` records the voxel-to-world affine of the grid, so a
mask is mapped onto it exactly by world coordinates instead of being interpolated as if the
two arrays shared axis order and field of view (which they do not).
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelGrid:
    """Voxel grid of a model input: affine (voxel -> world mm), shape and supersampling.

    ``supersample`` > 1 means every grid cell covers that many finer voxels; masks are then
    reported as the covered fraction of each cell, so thin structures are not lost.
    """

    affine: np.ndarray
    shape: tuple[int, int, int]
    supersample: tuple[int, int, int] = (1, 1, 1)

    def signature(self) -> str:
        payload = json.dumps(
            {
                "affine": np.round(self.affine, 6).tolist(),
                "shape": list(self.shape),
                "supersample": list(self.supersample),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def sidecar_path(volume_path: str | Path) -> Path:
    path = Path(volume_path)
    return path.with_name(path.name + ".metadata.json")


def read_sidecar(volume_path: str | Path) -> dict[str, Any] | None:
    path = sidecar_path(volume_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def model_grid(metadata: Mapping[str, Any] | None) -> ModelGrid | None:
    """The grid a cached tensor lives on, or None when the sidecar does not describe one."""
    if not metadata:
        return None
    feature = metadata.get("feature_grid")
    if isinstance(feature, Mapping) and feature.get("affine") and feature.get("shape"):
        return ModelGrid(
            np.asarray(feature["affine"], dtype=float),
            tuple(int(value) for value in feature["shape"]),
            tuple(int(value) for value in feature.get("cell_voxels") or (1, 1, 1)),
        )
    if metadata.get("output_affine") and metadata.get("output_shape"):
        shape = tuple(int(value) for value in metadata["output_shape"])[-3:]
        return ModelGrid(np.asarray(metadata["output_affine"], dtype=float), shape)
    return None


def _fine_affine(grid: ModelGrid) -> np.ndarray:
    """Affine of the supersampled grid: fine voxel u sits at coarse (u + 0.5) / f - 0.5."""
    factors = np.asarray(grid.supersample, dtype=float)
    scale = np.eye(4)
    scale[:3, :3] = np.diag(1.0 / factors)
    scale[:3, 3] = 0.5 / factors - 0.5
    return grid.affine @ scale


def mask_on_grid(mask: np.ndarray, mask_affine: np.ndarray, grid: ModelGrid) -> np.ndarray:
    """Nearest-neighbour resampling of a native mask onto ``grid`` (cell fraction if supersampled)."""
    from scipy import ndimage

    binary = np.asarray(mask) > 0
    factors = tuple(int(value) for value in grid.supersample)
    fine_shape = tuple(size * factor for size, factor in zip(grid.shape, factors))
    # fine voxel index -> world mm -> native mask index
    mapping = np.linalg.inv(np.asarray(mask_affine, dtype=float)) @ _fine_affine(grid)
    fine = ndimage.affine_transform(
        binary.astype(np.uint8),
        mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=fine_shape,
        order=0,
        mode="constant",
        cval=0,
    )
    if factors == (1, 1, 1):
        return fine.astype(np.float32)
    blocks = fine.reshape(
        grid.shape[0], factors[0], grid.shape[1], factors[1], grid.shape[2], factors[2]
    )
    return blocks.mean(axis=(1, 3, 5), dtype=np.float32)


def aligned_mask(
    mask_path: str | Path,
    grid: ModelGrid,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """``mask_on_grid`` for a NIfTI file, cached on disk per (mask file, grid)."""
    import nibabel as nib

    source = Path(mask_path)
    cache_file: Path | None = None
    if cache_dir is not None:
        stat = source.stat()
        key = hashlib.sha1(
            f"{source.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{grid.signature()}".encode("utf-8")
        ).hexdigest()[:20]
        cache_file = Path(cache_dir) / f"{key}.npy"
        if cache_file.is_file():
            try:
                return np.load(cache_file).astype(np.float32)
            except (OSError, ValueError):
                pass
    image = nib.load(str(source))
    result = mask_on_grid(np.asarray(image.dataobj), np.asarray(image.affine, dtype=float), grid)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(f".{cache_file.stem}.{uuid.uuid4().hex}.tmp.npy")
        try:
            np.save(temporary, result.astype(np.float16))
            os.replace(temporary, cache_file)
        except OSError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    return result


def cell_fraction(valid: np.ndarray, cell: Sequence[int]) -> np.ndarray:
    """Mean of ``valid`` over non-overlapping cells of size ``cell`` (shape must divide)."""
    factors = tuple(int(value) for value in cell)
    shape = tuple(size // factor for size, factor in zip(valid.shape, factors))
    blocks = np.asarray(valid, dtype=np.float32).reshape(
        shape[0], factors[0], shape[1], factors[1], shape[2], factors[2]
    )
    return blocks.mean(axis=(1, 3, 5))


__all__ = [
    "ModelGrid",
    "aligned_mask",
    "cell_fraction",
    "mask_on_grid",
    "model_grid",
    "read_sidecar",
    "sidecar_path",
]
