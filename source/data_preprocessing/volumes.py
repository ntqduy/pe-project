"""Physical-space CT preprocessing shared by every INSPECT dataset profile.

The original project cache was ported from a small debugging script: it clipped
HU values and centre-cropped/padded them into a fixed NPY tensor.  That suited
early dense models, but silently lost anatomy, affine geometry and robust cache
provenance.  This implementation keeps the NPY interface used by current
loaders while making the physical transformation and its QC explicit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SHAPE = (128, 128, 128)
PREPROCESSING_IMPLEMENTATION = "ctpa-physical-v2"
_FOREGROUND_STRATEGIES = {"none", "external_body"}
_SPATIAL_STRATEGIES = {"centre_crop_or_pad", "fit"}


@dataclass(frozen=True)
class PreprocessingSpec:
    """The versioned, hashable contract for one CTPA model-input cache entry.

    ``fit`` is recommended for dense models: after optional physical resampling
    and body cropping, it maps the complete retained field of view to the fixed
    tensor shape.  This prevents the pathology loss caused by a centre crop.
    Its per-case effective spacing remains available in the metadata sidecar.
    """

    target_shape: tuple[int, int, int] = DEFAULT_SHAPE
    clip_min_hu: float = -1000.0
    clip_max_hu: float = 1000.0
    normalize: str = "minmax"  # minmax | zscore | none
    dtype: str = "float32"
    orientation: str | None = "RAS"
    resample_spacing_mm: tuple[float, float, float] | None = None
    foreground_strategy: str = "none"  # none | external_body
    foreground_threshold_hu: float = -900.0
    foreground_margin_mm: float = 10.0
    spatial_strategy: str = "centre_crop_or_pad"  # centre_crop_or_pad | fit
    output_format: str = "npy"  # npy | pt
    emit_patch_grid: bool = False
    patch_shape: tuple[int, int, int] = (64, 64, 64)
    patch_overlap: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.target_shape) != 3 or any(int(value) < 1 for value in self.target_shape):
            raise ValueError("preprocessing.target_shape must be three positive integers")
        if self.clip_min_hu >= self.clip_max_hu:
            raise ValueError("preprocessing.clip_min_hu must be below clip_max_hu")
        if self.normalize not in {"minmax", "zscore", "none"}:
            raise ValueError("preprocessing.normalize must be minmax, zscore or none")
        if self.output_format not in {"npy", "pt"}:
            raise ValueError("preprocessing.output_format must be npy or pt")
        if self.orientation is not None:
            orientation = self.orientation.upper()
            pairs = ({"L", "R"}, {"P", "A"}, {"I", "S"})
            if len(orientation) != 3 or any(len(set(orientation) & pair) != 1 for pair in pairs):
                raise ValueError("preprocessing.orientation must have one LR, PA and IS axis")
        if self.resample_spacing_mm and (
            len(self.resample_spacing_mm) != 3 or any(value <= 0 for value in self.resample_spacing_mm)
        ):
            raise ValueError("preprocessing.resample_spacing_mm must be three positive values")
        if self.foreground_strategy not in _FOREGROUND_STRATEGIES:
            raise ValueError("preprocessing.foreground_strategy must be none or external_body")
        if self.foreground_margin_mm < 0:
            raise ValueError("preprocessing.foreground_margin_mm must be non-negative")
        if self.spatial_strategy not in _SPATIAL_STRATEGIES:
            raise ValueError("preprocessing.spatial_strategy must be centre_crop_or_pad or fit")
        if not 0 <= self.patch_overlap < 1:
            raise ValueError("preprocessing.patch_overlap must be in [0, 1)")
        if len(self.patch_shape) != 3 or any(value < 1 for value in self.patch_shape):
            raise ValueError("preprocessing.patch_shape must be three positive integers")
        if self.emit_patch_grid and any(patch > target for patch, target in zip(self.patch_shape, self.target_shape)):
            raise ValueError("preprocessing.patch_shape cannot exceed target_shape")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PreprocessingSpec":
        data = dict(payload or {})
        shape = data.pop("target_shape", DEFAULT_SHAPE)
        spacing = data.pop("resample_spacing_mm", None)
        patch_shape = data.pop("patch_shape", (64, 64, 64))
        known = {
            "clip_min_hu", "clip_max_hu", "normalize", "dtype", "orientation",
            "foreground_strategy", "foreground_threshold_hu", "foreground_margin_mm",
            "spatial_strategy", "output_format", "emit_patch_grid", "patch_overlap",
        }
        extra = {key: value for key, value in data.items() if key not in known}
        return cls(
            target_shape=tuple(int(value) for value in shape),  # type: ignore[arg-type]
            clip_min_hu=float(data.get("clip_min_hu", -1000.0)),
            clip_max_hu=float(data.get("clip_max_hu", 1000.0)),
            normalize=str(data.get("normalize", "minmax")),
            dtype=str(data.get("dtype", "float32")),
            orientation=str(data["orientation"]).upper() if data.get("orientation") else None,
            resample_spacing_mm=tuple(float(value) for value in spacing) if spacing else None,
            foreground_strategy=str(data.get("foreground_strategy", "none")),
            foreground_threshold_hu=float(data.get("foreground_threshold_hu", -900.0)),
            foreground_margin_mm=float(data.get("foreground_margin_mm", 10.0)),
            spatial_strategy=str(data.get("spatial_strategy", "centre_crop_or_pad")),
            output_format=str(data.get("output_format", "npy")),
            emit_patch_grid=bool(data.get("emit_patch_grid", False)),
            patch_shape=tuple(int(value) for value in patch_shape),  # type: ignore[arg-type]
            patch_overlap=float(data.get("patch_overlap", 0.5)),
            extra=extra,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("target_shape", "resample_spacing_mm", "patch_shape"):
            if payload[name] is not None:
                payload[name] = list(payload[name])
        payload["implementation"] = PREPROCESSING_IMPLEMENTATION
        return payload

    def fingerprint(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def suffix(self) -> str:
        return ".npy" if self.output_format == "npy" else ".pt"


@dataclass(frozen=True)
class PatchCoordinate:
    patch_id: int
    z: int
    y: int
    x: int
    dz: int
    dy: int
    dx: int
    world_x_mm: float
    world_y_mm: float
    world_z_mm: float


def axis_starts(length: int, patch: int, overlap: float = 0.5) -> list[int]:
    if length < 1 or patch < 1 or patch > length or not 0 <= overlap < 1:
        raise ValueError("invalid patch-grid dimensions or overlap")
    if length == patch:
        return [0]
    step = max(1, int(round(patch * (1 - overlap))))
    starts = list(range(0, length - patch + 1, step))
    final = length - patch
    if starts[-1] != final:
        starts.append(final)
    return starts


def patch_grid(shape: Sequence[int], patch: Sequence[int], overlap: float, affine: Any) -> list[PatchCoordinate]:
    """Generate a deterministic complete cover and world-coordinate patch centres."""
    import numpy as np

    shape_tuple = tuple(int(value) for value in shape)
    patch_tuple = tuple(int(value) for value in patch)
    matrix = np.asarray(affine, dtype=float)
    coordinates: list[PatchCoordinate] = []
    patch_id = 0
    for z in axis_starts(shape_tuple[0], patch_tuple[0], overlap):
        for y in axis_starts(shape_tuple[1], patch_tuple[1], overlap):
            for x in axis_starts(shape_tuple[2], patch_tuple[2], overlap):
                centre = matrix @ np.asarray([z + patch_tuple[0] / 2, y + patch_tuple[1] / 2, x + patch_tuple[2] / 2, 1])
                coordinates.append(PatchCoordinate(patch_id, z, y, x, *patch_tuple, *map(float, centre[:3])))
                patch_id += 1
    return coordinates


def _require_imaging() -> tuple[Any, Any, Any]:
    try:
        import nibabel as nib
        import numpy as np
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel, numpy and scipy are required for CT preprocessing") from exc
    return nib, np, ndimage


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _affine_with_spacing(affine: Any, old_spacing: Sequence[float], new_spacing: Sequence[float]) -> Any:
    _, np, _ = _require_imaging()
    result = np.asarray(affine, dtype=float).copy()
    directions = result[:3, :3] / np.asarray(old_spacing, dtype=float)[None, :]
    result[:3, :3] = directions * np.asarray(new_spacing, dtype=float)[None, :]
    return result


def center_crop_or_pad(volume: Any, target_shape: Sequence[int]) -> Any:
    """Legacy public helper: centre-crop/zero-pad without geometry metadata."""
    _, np, _ = _require_imaging()
    output = np.zeros(tuple(int(value) for value in target_shape), dtype=volume.dtype)
    source_slices, target_slices = [], []
    for source_size, target_size in zip(volume.shape, target_shape):
        take = min(int(source_size), int(target_size))
        source_start = max(0, (int(source_size) - take) // 2)
        target_start = max(0, (int(target_size) - take) // 2)
        source_slices.append(slice(source_start, source_start + take))
        target_slices.append(slice(target_start, target_start + take))
    output[tuple(target_slices)] = volume[tuple(source_slices)]
    return output


def _centre_crop_or_pad_with_affine(volume: Any, affine: Any, target_shape: Sequence[int], pad_value: float) -> tuple[Any, Any, list[list[int]], list[list[int]]]:
    _, np, _ = _require_imaging()
    output = np.full(tuple(int(value) for value in target_shape), pad_value, dtype=volume.dtype)
    source_slices, target_slices, offsets = [], [], []
    for source_size, target_size in zip(volume.shape, target_shape):
        take = min(int(source_size), int(target_size))
        source_start = max(0, (int(source_size) - take) // 2)
        target_start = max(0, (int(target_size) - take) // 2)
        source_slices.append(slice(source_start, source_start + take))
        target_slices.append(slice(target_start, target_start + take))
        offsets.append(source_start - target_start)
    output[tuple(target_slices)] = volume[tuple(source_slices)]
    result_affine = np.asarray(affine, dtype=float).copy()
    result_affine[:3, 3] = (np.asarray(affine, dtype=float) @ np.asarray([*offsets, 1]))[:3]
    return output, result_affine, [[item.start, item.stop] for item in source_slices], [[item.start, item.stop] for item in target_slices]


def _crop_external_body(volume: Any, affine: Any, spacing: Sequence[float], spec: PreprocessingSpec) -> tuple[Any, Any, dict[str, Any]]:
    _, np, ndimage = _require_imaging()
    if spec.foreground_strategy == "none":
        return volume, affine, {
            "foreground_strategy": "none", "foreground_empty": False,
            "crop_box": [0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2]],
            "crop_fraction": 1.0, "cropped_shape_before_fit": list(volume.shape),
        }
    foreground = volume > spec.foreground_threshold_hu
    labels, components = ndimage.label(foreground)
    if not components:
        return volume, affine, {
            "foreground_strategy": spec.foreground_strategy, "foreground_empty": True,
            "crop_box": [0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2]],
            "crop_fraction": 1.0, "cropped_shape_before_fit": list(volume.shape),
        }
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    points = np.argwhere(labels == int(sizes.argmax()))
    margin = np.asarray([max(0, int(round(spec.foreground_margin_mm / value))) for value in spacing])
    lower = np.maximum(0, points.min(axis=0) - margin)
    upper = np.minimum(np.asarray(volume.shape), points.max(axis=0) + 1 + margin)
    cropped = volume[tuple(slice(int(low), int(high)) for low, high in zip(lower, upper))]
    cropped_affine = np.asarray(affine, dtype=float).copy()
    cropped_affine[:3, 3] = (np.asarray(affine, dtype=float) @ np.asarray([*lower, 1]))[:3]
    return cropped, cropped_affine, {
        "foreground_strategy": spec.foreground_strategy, "foreground_empty": False,
        "crop_box": [int(value) for pair in zip(lower, upper) for value in pair],
        "crop_fraction": float(math.prod(cropped.shape) / max(1, math.prod(volume.shape))),
        "cropped_shape_before_fit": list(cropped.shape),
    }


def _fit_to_shape(volume: Any, affine: Any, spacing: Sequence[float], target_shape: Sequence[int]) -> tuple[Any, Any, tuple[float, float, float]]:
    _, np, ndimage = _require_imaging()
    target = tuple(int(value) for value in target_shape)
    source_shape = tuple(int(value) for value in volume.shape)
    if source_shape == target:
        return volume, np.asarray(affine, dtype=float).copy(), tuple(float(value) for value in spacing)
    factors = tuple(destination / source for destination, source in zip(target, source_shape))
    fitted = ndimage.zoom(volume, factors, order=1, mode="constant", cval=float(volume.min()), prefilter=False)
    # scipy honours the requested shape in supported versions. Repair a rounding edge case
    # only; this is not the intentional destructive crop of the legacy pipeline.
    if tuple(fitted.shape) != target:
        fitted = center_crop_or_pad(fitted, target)
    effective_spacing = tuple(float(old * source / destination) for old, source, destination in zip(spacing, source_shape, target))
    return fitted, _affine_with_spacing(affine, spacing, effective_spacing), effective_spacing


def preprocess_volume_with_metadata(path: str | Path, spec: PreprocessingSpec) -> tuple[Any, dict[str, Any]]:
    """Apply physical preprocessing and return a dense tensor with its provenance."""
    nib, np, ndimage = _require_imaging()
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"expected a 3-D CT, got {image.shape}")
    original_affine = np.asarray(image.affine, dtype=float).copy()
    original_shape = tuple(int(value) for value in image.shape)
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("volume contains NaN or Inf before preprocessing")
    original_spacing = tuple(float(value) for value in nib.affines.voxel_sizes(original_affine)[:3])

    oriented_affine = original_affine.copy()
    if spec.orientation:
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(original_affine),
            nib.orientations.axcodes2ornt(tuple(spec.orientation)),
        )
        data = nib.orientations.apply_orientation(data, transform)
        oriented_affine = original_affine @ nib.orientations.inv_ornt_aff(transform, original_shape)
    oriented_spacing = tuple(float(value) for value in nib.affines.voxel_sizes(oriented_affine)[:3])

    resampled_affine = oriented_affine.copy()
    spacing = oriented_spacing
    if spec.resample_spacing_mm:
        factors = tuple(old / new for old, new in zip(oriented_spacing, spec.resample_spacing_mm))
        data = ndimage.zoom(data, factors, order=1, mode="constant", cval=spec.clip_min_hu, prefilter=False).astype(np.float32, copy=False)
        spacing = tuple(float(value) for value in spec.resample_spacing_mm)
        resampled_affine = _affine_with_spacing(oriented_affine, oriented_spacing, spacing)
    resampled_shape = tuple(int(value) for value in data.shape)
    data = np.clip(data, spec.clip_min_hu, spec.clip_max_hu)
    data, cropped_affine, crop_metadata = _crop_external_body(data, resampled_affine, spacing, spec)

    spatial_metadata: dict[str, Any] = {"spatial_strategy": spec.spatial_strategy}
    if spec.spatial_strategy == "fit":
        data, output_affine, output_spacing = _fit_to_shape(data, cropped_affine, spacing, spec.target_shape)
        spatial_metadata["source_bounds"] = [[0, int(value)] for value in crop_metadata["cropped_shape_before_fit"]]
        spatial_metadata["target_bounds"] = [[0, int(value)] for value in spec.target_shape]
    else:
        data, output_affine, source_bounds, target_bounds = _centre_crop_or_pad_with_affine(data, cropped_affine, spec.target_shape, spec.clip_min_hu)
        output_spacing = spacing
        spatial_metadata["source_bounds"] = source_bounds
        spatial_metadata["target_bounds"] = target_bounds
    if not np.isfinite(data).all():
        raise ValueError("volume contains NaN or Inf after geometric preprocessing")
    hu_quantiles = {str(value): float(np.quantile(data, value)) for value in (0.01, 0.1, 0.5, 0.9, 0.99)}
    if spec.normalize == "minmax":
        data = (data - spec.clip_min_hu) / (spec.clip_max_hu - spec.clip_min_hu)
    elif spec.normalize == "zscore":
        deviation = float(data.std())
        data = (data - float(data.mean())) / (deviation if deviation > 0 else 1.0)
    output = data.astype(spec.dtype, copy=False)
    return output, {
        "schema_version": 2,
        "preprocessing_implementation": PREPROCESSING_IMPLEMENTATION,
        "original_shape": list(original_shape), "original_affine": original_affine.tolist(),
        "original_spacing_mm": list(original_spacing), "oriented_affine": oriented_affine.tolist(),
        "oriented_spacing_mm": list(oriented_spacing), "resampled_affine": resampled_affine.tolist(),
        "resampled_shape_before_crop": list(resampled_shape), "output_affine": output_affine.tolist(),
        "output_spacing_mm": list(output_spacing), "output_shape": list(output.shape),
        "orientation": spec.orientation or "native", "hu_range": [spec.clip_min_hu, spec.clip_max_hu],
        "hu_quantiles_before_normalization": hu_quantiles, "normalization": spec.normalize,
        "dtype": str(output.dtype),
        "physical_field_of_view_mm": [float(length * voxel) for length, voxel in zip(output.shape, output_spacing)],
        **crop_metadata, **spatial_metadata,
    }


def preprocess_volume(path: str | Path, spec: PreprocessingSpec) -> Any:
    """Compatibility entry point returning only the model-input tensor."""
    return preprocess_volume_with_metadata(path, spec)[0]


def _sidecar_paths(destination: Path) -> tuple[Path, Path]:
    return destination.with_suffix(destination.suffix + ".metadata.json"), destination.with_suffix(destination.suffix + ".patches.csv")


def _write_patch_grid(path: Path, coordinates: Sequence[PatchCoordinate]) -> str:
    if not coordinates:
        raise ValueError("patch grid is empty")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(coordinates[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in coordinates)
    os.replace(temporary, path)
    return _sha256_file(path)


def preprocess_study(study_id: str, source_path: str | Path, output_dir: str | Path, spec: PreprocessingSpec, *, overwrite: bool = False) -> dict[str, Any]:
    """Atomically write one derivative, or reuse it only if source and contract match."""
    import numpy as np

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"CT source does not exist: {source}")
    destination = Path(output_dir) / f"{study_id}{spec.suffix()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path, patch_path = _sidecar_paths(destination)
    fingerprint, source_sha256 = spec.fingerprint(), _sha256_file(source)
    if destination.is_file() and metadata_path.is_file() and not overwrite:
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cache sidecar is unreadable: {metadata_path}") from exc
        same = (
            existing.get("preprocessing_fingerprint") == fingerprint
            and existing.get("preprocessing_implementation") == PREPROCESSING_IMPLEMENTATION
            and existing.get("source_path") == str(source)
            and existing.get("source_sha256") == source_sha256
            and (not spec.emit_patch_grid or patch_path.is_file())
        )
        if not same:
            raise ValueError(f"cache entry for {study_id} has a different source or preprocessing contract; pass overwrite=True")
        return {
            "study_id": study_id, "source_path": str(source), "preprocessed_path": str(destination),
            "metadata_path": str(metadata_path), "patch_manifest": str(patch_path) if spec.emit_patch_grid else "",
            "shape": existing.get("output_shape"), "status": "cached", "preprocessing_fingerprint": fingerprint,
        }

    output, metadata = preprocess_volume_with_metadata(source, spec)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        if spec.output_format == "npy":
            with temporary.open("wb") as handle:
                np.save(handle, output)
        else:
            import torch
            torch.save(torch.from_numpy(output), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    patch_sha256 = ""
    if spec.emit_patch_grid:
        coordinates = patch_grid(output.shape, spec.patch_shape, spec.patch_overlap, metadata["output_affine"])
        patch_sha256 = _write_patch_grid(patch_path, coordinates)
        metadata.update({"patch_manifest": str(patch_path), "patch_manifest_sha256": patch_sha256, "patch_count": len(coordinates)})
    else:
        metadata["patch_count"] = 0
    metadata.update({
        "study_id": study_id, "source_path": str(source), "source_sha256": source_sha256,
        "preprocessing_fingerprint": fingerprint, "status": "completed",
    })
    _atomic_json(metadata_path, metadata)
    return {
        "study_id": study_id, "source_path": str(source), "preprocessed_path": str(destination),
        "metadata_path": str(metadata_path), "patch_manifest": str(patch_path) if spec.emit_patch_grid else "",
        "shape": metadata["output_shape"], "status": "written", "preprocessing_fingerprint": fingerprint,
        "patch_manifest_sha256": patch_sha256,
    }


def validate_cache_entry(path: str | Path, spec: PreprocessingSpec) -> list[str]:
    """Validate tensor, physical sidecar and optional patch manifest before inclusion."""
    import numpy as np

    source = Path(path)
    metadata_path, patch_path = _sidecar_paths(source)
    errors: list[str] = []
    if not source.is_file():
        return [f"missing preprocessed file: {source}"]
    try:
        array = np.load(source, mmap_mode="r") if spec.output_format == "npy" else None
        if array is None:
            import torch
            array = torch.load(source, map_location="cpu", weights_only=False).numpy()
        if tuple(array.shape) != spec.target_shape:
            errors.append(f"{source}: shape {tuple(array.shape)} != {spec.target_shape}")
        if str(array.dtype) != spec.dtype:
            errors.append(f"{source}: dtype {array.dtype} != {spec.dtype}")
        if not np.isfinite(array).all():
            errors.append(f"{source}: contains NaN or Inf")
        elif spec.normalize == "minmax" and (float(array.min()) < -1e-6 or float(array.max()) > 1 + 1e-6):
            errors.append(f"{source}: values outside normalized [0, 1] range")
    except Exception as exc:  # noqa: BLE001 - batch QC must report and continue
        return [f"{source}: {type(exc).__name__}: {exc}"]
    if not metadata_path.is_file():
        return [*errors, f"{source}: missing preprocessing sidecar {metadata_path.name}"]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [*errors, f"{source}: unreadable preprocessing sidecar: {exc}"]
    if metadata.get("status") != "completed":
        errors.append(f"{source}: preprocessing sidecar is not completed")
    if metadata.get("preprocessing_fingerprint") != spec.fingerprint():
        errors.append(f"{source}: preprocessing fingerprint mismatch")
    if metadata.get("preprocessing_implementation") != PREPROCESSING_IMPLEMENTATION:
        errors.append(f"{source}: preprocessing implementation mismatch")
    if len(str(metadata.get("source_sha256") or "")) != 64:
        errors.append(f"{source}: missing source SHA-256 provenance")
    affine = np.asarray(metadata.get("output_affine"), dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        errors.append(f"{source}: invalid output affine")
    spacing = np.asarray(metadata.get("output_spacing_mm"), dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        errors.append(f"{source}: invalid output spacing")
    try:
        if not 0 < float(metadata.get("crop_fraction")) <= 1:
            errors.append(f"{source}: invalid crop fraction")
    except (TypeError, ValueError):
        errors.append(f"{source}: invalid crop fraction")
    if spec.emit_patch_grid:
        if not patch_path.is_file():
            errors.append(f"{source}: missing patch manifest")
        elif metadata.get("patch_manifest_sha256") != _sha256_file(patch_path):
            errors.append(f"{source}: patch manifest hash mismatch")
        elif int(metadata.get("patch_count") or 0) < 1:
            errors.append(f"{source}: empty patch manifest")
    return errors
