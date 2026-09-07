"""CT volume preprocessing -- one implementation, shared by every dataset profile.

Ported from /mnt/pe_study `preprocess_ct_sample.py` (HU clipping, intensity scaling,
centre crop/pad to a fixed shape) and `validate_ct_cache.py` (post-hoc cache QC),
with the pilot-only assumptions removed: nothing here knows whether it is running on
500 patients or on the whole release.

`test_500_sample` and `full_inspect` MUST resolve to the same `PreprocessingSpec`;
`source/dataset/profiles/_common.yaml` is what guarantees that, and
`pipeline.build_dataset` records the spec hash in every dataset's provenance so a
mismatch is visible in the manifest rather than only in the results.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SHAPE = (128, 128, 128)


@dataclass(frozen=True)
class PreprocessingSpec:
    """The complete, hashable contract for turning one CTPA into a model input."""

    target_shape: tuple[int, int, int] = DEFAULT_SHAPE
    clip_min_hu: float = -1000.0
    clip_max_hu: float = 1000.0
    normalize: str = "minmax"       # minmax | zscore | none
    dtype: str = "float32"
    orientation: str | None = "RAS"
    resample_spacing_mm: tuple[float, float, float] | None = None
    output_format: str = "npy"      # npy | pt
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.target_shape) != 3 or any(int(value) < 1 for value in self.target_shape):
            raise ValueError("preprocessing.target_shape must be three positive integers")
        if float(self.clip_min_hu) >= float(self.clip_max_hu):
            raise ValueError("preprocessing.clip_min_hu must be below clip_max_hu")
        if self.normalize not in {"minmax", "zscore", "none"}:
            raise ValueError("preprocessing.normalize must be minmax, zscore or none")
        if self.output_format not in {"npy", "pt"}:
            raise ValueError("preprocessing.output_format must be npy or pt")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> PreprocessingSpec:
        data = dict(payload or {})
        shape = data.pop("target_shape", DEFAULT_SHAPE)
        spacing = data.pop("resample_spacing_mm", None)
        known = {
            "clip_min_hu",
            "clip_max_hu",
            "normalize",
            "dtype",
            "orientation",
            "output_format",
        }
        extra = {key: value for key, value in data.items() if key not in known}
        return cls(
            target_shape=tuple(int(value) for value in shape),  # type: ignore[arg-type]
            clip_min_hu=float(data.get("clip_min_hu", -1000.0)),
            clip_max_hu=float(data.get("clip_max_hu", 1000.0)),
            normalize=str(data.get("normalize", "minmax")),
            dtype=str(data.get("dtype", "float32")),
            orientation=data.get("orientation", "RAS"),
            resample_spacing_mm=tuple(float(value) for value in spacing) if spacing else None,
            output_format=str(data.get("output_format", "npy")),
            extra=extra,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_shape"] = list(self.target_shape)
        if self.resample_spacing_mm:
            payload["resample_spacing_mm"] = list(self.resample_spacing_mm)
        return payload

    def fingerprint(self) -> str:
        """Stable hash of the preprocessing contract, recorded in dataset provenance."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def suffix(self) -> str:
        return ".npy" if self.output_format == "npy" else ".pt"


def center_crop_or_pad(volume: Any, target_shape: Sequence[int]) -> Any:
    """Centre-crop or zero-pad each axis to ``target_shape`` without resampling."""
    import numpy as np

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


def preprocess_volume(path: str | Path, spec: PreprocessingSpec) -> Any:
    """Load one CTPA and apply the shared preprocessing contract. Returns a NumPy array."""
    import nibabel as nib
    import numpy as np

    image = nib.load(str(path))
    if spec.orientation:
        try:
            image = nib.as_closest_canonical(image)
        except Exception:  # noqa: BLE001 - a missing/odd affine must not abort the case
            pass
    volume = np.asanyarray(image.dataobj).astype("float32")
    if not np.isfinite(volume).all():
        raise ValueError("volume contains NaN or Inf before preprocessing")
    if spec.resample_spacing_mm:
        volume = _resample(volume, image, spec.resample_spacing_mm)
    volume = np.clip(volume, spec.clip_min_hu, spec.clip_max_hu)
    if spec.normalize == "minmax":
        volume = (volume - spec.clip_min_hu) / (spec.clip_max_hu - spec.clip_min_hu)
    elif spec.normalize == "zscore":
        deviation = float(volume.std())
        volume = (volume - float(volume.mean())) / (deviation if deviation > 0 else 1.0)
    volume = center_crop_or_pad(volume, spec.target_shape)
    return volume.astype(spec.dtype, copy=False)


def _resample(volume: Any, image: Any, spacing: Sequence[float]) -> Any:
    """Resample to an isotropic-ish target spacing using the header zooms."""
    import numpy as np
    from scipy import ndimage  # imported lazily: only needed when resampling is configured

    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if any(value <= 0 for value in zooms):
        raise ValueError("cannot resample: the NIfTI header declares a non-positive voxel size")
    factors = [zoom / float(target) for zoom, target in zip(zooms, spacing)]
    return ndimage.zoom(volume, factors, order=1).astype(np.float32, copy=False)


def preprocess_study(
    study_id: str,
    source_path: str | Path,
    output_dir: str | Path,
    spec: PreprocessingSpec,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Preprocess one study into the cache. Returns the cache manifest row."""
    import numpy as np

    destination = Path(output_dir) / f"{study_id}{spec.suffix()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not overwrite:
        existing = np.load(destination, mmap_mode="r") if spec.output_format == "npy" else None
        shape = tuple(int(value) for value in existing.shape) if existing is not None else None
        return {
            "study_id": study_id,
            "source_path": str(source_path),
            "preprocessed_path": str(destination),
            "shape": list(shape) if shape else None,
            "status": "cached",
            "preprocessing_fingerprint": spec.fingerprint(),
        }
    volume = preprocess_volume(source_path, spec)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if spec.output_format == "npy":
        np.save(temporary, volume)
        temporary.replace(destination)
    else:
        import torch

        torch.save(torch.from_numpy(volume), temporary)
        temporary.replace(destination)
    return {
        "study_id": study_id,
        "source_path": str(source_path),
        "preprocessed_path": str(destination),
        "shape": list(volume.shape),
        "status": "written",
        "preprocessing_fingerprint": spec.fingerprint(),
    }


def validate_cache_entry(path: str | Path, spec: PreprocessingSpec) -> list[str]:
    """Post-hoc QC of one cached volume against the spec that should have produced it."""
    import numpy as np

    source = Path(path)
    if not source.is_file():
        return [f"missing preprocessed file: {source}"]
    errors: list[str] = []
    try:
        array = np.load(source, mmap_mode="r") if spec.output_format == "npy" else None
        if array is None:
            import torch

            array = torch.load(source, map_location="cpu", weights_only=False).numpy()
        if tuple(array.shape) != tuple(spec.target_shape):
            errors.append(f"{source}: shape {tuple(array.shape)} != {tuple(spec.target_shape)}")
        if str(array.dtype) != spec.dtype:
            errors.append(f"{source}: dtype {array.dtype} != {spec.dtype}")
        if not np.isfinite(array).all():
            errors.append(f"{source}: contains NaN or Inf")
        elif spec.normalize == "minmax":
            low, high = float(array.min()), float(array.max())
            if low < -1e-6 or high > 1 + 1e-6:
                errors.append(f"{source}: range [{low}, {high}] outside [0, 1]")
    except Exception as exc:  # noqa: BLE001 - QC reports, never raises
        errors.append(f"{source}: {type(exc).__name__}: {exc}")
    return errors
