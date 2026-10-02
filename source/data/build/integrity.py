"""Corrupted / missing CT detection.

Generalized from the failure handling in /mnt/pe_study `preprocess_ct_sample.py`
(per-case try/except into `failures.jsonl`) and `validate_ct_cache.py` (shape, dtype,
finiteness and range checks), lifted out of those batch scripts so both dataset
profiles and preflight can call the same check.

Two levels, because they cost very different amounts:

``level="path"``   the file exists and is non-trivially sized. Metadata-speed.
``level="header"`` the NIfTI header parses and declares a plausible 3-D volume.
``level="volume"`` the voxel data actually decodes and is finite. Slow; opt in.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sources import StudyRecord

LEVELS = ("path", "header", "volume")
MINIMUM_FILE_BYTES = 1024


@dataclass(frozen=True)
class VolumeCheck:
    patient_id: str
    study_id: str
    path: str
    status: str          # PASS | MISSING | CORRUPT
    detail: str
    shape: tuple[int, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "path": self.path,
            "status": self.status,
            "detail": self.detail,
            "shape": list(self.shape) if self.shape else None,
        }


def _load_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "nibabel is required for header/volume integrity checks; "
            "use level='path' for a metadata-only pass"
        ) from exc
    return nib


def check_volume(
    record: StudyRecord,
    *,
    level: str = "header",
    minimum_dimensions: int = 3,
    minimum_slices: int = 2,
) -> VolumeCheck:
    if level not in LEVELS:
        raise ValueError(f"integrity level must be one of {LEVELS}")
    path = Path(record.image_path) if record.image_path else None
    if path is None or not path.is_file():
        return VolumeCheck(record.patient_id, record.study_id, str(path or ""), "MISSING", "file not found")
    size = path.stat().st_size
    if size < MINIMUM_FILE_BYTES:
        return VolumeCheck(
            record.patient_id, record.study_id, str(path), "CORRUPT", f"file is only {size} bytes"
        )
    if level == "path":
        return VolumeCheck(record.patient_id, record.study_id, str(path), "PASS", f"{size} bytes")

    nib = _load_nibabel()
    try:
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a corrupt volume
        return VolumeCheck(
            record.patient_id, record.study_id, str(path), "CORRUPT", f"{type(exc).__name__}: {exc}"
        )
    if len(shape) < minimum_dimensions or any(value < 1 for value in shape):
        return VolumeCheck(
            record.patient_id, record.study_id, str(path), "CORRUPT", f"implausible shape {shape}", shape
        )
    if minimum_slices and shape[-1] < minimum_slices and shape[0] < minimum_slices:
        return VolumeCheck(
            record.patient_id, record.study_id, str(path), "CORRUPT", f"too few slices: {shape}", shape
        )
    if level == "header":
        return VolumeCheck(record.patient_id, record.study_id, str(path), "PASS", f"shape={shape}", shape)

    try:
        import numpy as np

        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all():
            return VolumeCheck(
                record.patient_id, record.study_id, str(path), "CORRUPT", "volume contains NaN or Inf", shape
            )
        if float(data.min()) == float(data.max()):
            return VolumeCheck(
                record.patient_id, record.study_id, str(path), "CORRUPT", "volume is constant", shape
            )
    except Exception as exc:  # noqa: BLE001
        return VolumeCheck(
            record.patient_id, record.study_id, str(path), "CORRUPT", f"{type(exc).__name__}: {exc}", shape
        )
    return VolumeCheck(record.patient_id, record.study_id, str(path), "PASS", f"shape={shape}", shape)


def check_volumes(
    records: Sequence[StudyRecord],
    *,
    level: str = "header",
    limit: int | None = None,
    minimum_slices: int = 2,
) -> tuple[list[StudyRecord], list[VolumeCheck]]:
    """Return (usable records, every check). ``limit`` checks only the first N studies.

    Studies that were not checked because of ``limit`` are kept: a limited pass is a
    smoke test, and silently discarding unchecked studies would change the cohort.
    """
    checks: list[VolumeCheck] = []
    usable: list[StudyRecord] = []
    for index, record in enumerate(records):
        if limit is not None and index >= int(limit):
            usable.append(record)
            continue
        result = check_volume(record, level=level, minimum_slices=minimum_slices)
        checks.append(result)
        if result.status == "PASS":
            usable.append(record)
    return usable, checks


def integrity_summary(checks: Sequence[VolumeCheck]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "checked": len(checks),
        "by_status": dict(sorted(counts.items())),
        "failures": [check.as_dict() for check in checks if check.status != "PASS"][:200],
    }
