#!/usr/bin/env python
"""Build a CT-FM-compatible cache from an already-built official-split dataset.

This stage intentionally does not read the release split table to create a new split.
It reads the project's filtered manifests, verifies their ``train/validation/test``
assignments, and only changes the image path to a CT-FM-specific derivative.

The CT-FM repository's feature extractor uses SPL orientation, spacing [3, 1, 1],
foreground cropping and intensity scaling from [-1024, 2048] to [0, 1].  The current
training engine consumes one dense tensor per study, so this adapter stores a complete
single-patch [24, 128, 128] representation after foreground cropping.  The raw NIfTI
and the original project cache remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.manifests import read_rows
from source.data_preprocessing.volumes import (
    PreprocessingSpec,
    preprocess_study,
    validate_cache_entry,
)


MANIFESTS = {
    "diagnosis": "diagnosis.csv",
    "prognosis": "prognosis.csv",
    "prognosis_all_patient": "prognosis_all_patient.csv",
    "prognosis_pe_positive": "prognosis_pe_positive.csv",
}
PROGNOSIS_TARGETS = (
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or [])
    for row in rows:
        for field in row:
            if field not in columns:
                columns.append(field)
    if not columns:
        raise ValueError(f"cannot write an empty-schema CSV: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _release_root(raw_root: Path) -> Path:
    """Accept either the release root or the parent containing CT/full."""
    candidates = (raw_root, raw_root / "CT" / "full")
    for candidate in candidates:
        if (candidate / "CTPA").is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not find CTPA volumes below {raw_root}; expected <root>/CT/full/CTPA"
    )


def _validate_manifest(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"manifest is empty: {name}")
    required = {"patient_id", "study_id", "split", "image_path"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")
    splits = {str(row.get("split") or "").strip() for row in rows}
    invalid = sorted(splits - {"train", "validation", "test"})
    if invalid:
        raise ValueError(f"{name} has unsupported split values: {invalid}")
    patients: dict[str, set[str]] = defaultdict(set)
    pairs: list[tuple[str, str]] = []
    for row in rows:
        patient = str(row.get("patient_id") or "").strip()
        study = str(row.get("study_id") or "").strip()
        if not patient or not study:
            raise ValueError(f"{name} contains an empty patient_id/study_id")
        patients[patient].add(str(row["split"]).strip())
        pairs.append((patient, study))
    overlap = sorted(patient for patient, values in patients.items() if len(values) > 1)
    duplicates = sorted(pair for pair, count in Counter(pairs).items() if count > 1)
    if overlap:
        raise ValueError(f"{name} has patients in multiple splits: {overlap[:5]}")
    if duplicates:
        raise ValueError(f"{name} has duplicate patient/study rows: {duplicates[:5]}")
    return {
        "rows": len(rows),
        "patients": len(patients),
        "split_rows": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "split_patients": {
            split: len({str(row["patient_id"]) for row in rows if str(row["split"]) == split})
            for split in ("train", "validation", "test")
        },
    }


def _validate_labels(name: str, rows: list[dict[str, Any]]) -> None:
    columns = set(rows[0])
    if name == "diagnosis":
        required = {"pe_present", "pe_positive", "pe_acute", "pe_subsegmental_only"}
    elif name.startswith("prognosis"):
        required = {
            target
            for target in PROGNOSIS_TARGETS
        } | {
            f"{target}_status"
            for target in PROGNOSIS_TARGETS
        }
    else:
        return
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{name} is missing INSPECT label columns: {', '.join(missing)}")


def _raw_volume(release_root: Path, study_id: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        candidate = release_root / "CTPA" / f"{study_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"raw CT volume not found for study_id={study_id}")


def _spec() -> PreprocessingSpec:
    return PreprocessingSpec(
        target_shape=(24, 128, 128),
        clip_min_hu=-1024.0,
        clip_max_hu=2048.0,
        normalize="minmax",
        dtype="float32",
        orientation="SPL",
        resample_spacing_mm=(3.0, 1.0, 1.0),
        foreground_strategy="external_body",
        foreground_threshold_hu=0.0,
        foreground_margin_mm=0.0,
        spatial_strategy="fit",
        output_format="npy",
        emit_patch_grid=False,
        extra={
            "foundation_model": "CT-FM",
            "upstream_contract": "SPL; spacing=[3,1,1]; CropForeground; ScaleIntensityRanged[-1024,2048]",
            "mode": "dense_single_patch",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="derived/datasets/<profile> containing the existing manifests")
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="INSPECT root or directory containing CT/full/CTPA")
    parser.add_argument("--output-name", default="ct_fm_frozen")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    release_root = _release_root(args.raw_root.resolve())
    source_manifest_dir = dataset_root / "manifests"
    output_root = dataset_root / args.output_name
    output_manifest_dir = output_root / "manifests"
    cache_dir = output_root / "volumes"
    if output_root.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output_root}; pass --overwrite to rebuild it")
    output_root.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, list[dict[str, Any]]] = {}
    input_audit: dict[str, Any] = {}
    for name, filename in MANIFESTS.items():
        path = source_manifest_dir / filename
        rows = read_rows(path)
        input_audit[name] = _validate_manifest(name, rows)
        _validate_labels(name, rows)
        loaded[name] = rows

    spec = _spec()
    study_ids = sorted({str(row["study_id"]) for rows in loaded.values() for row in rows})
    cache_rows: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for study_id in study_ids:
        try:
            result = preprocess_study(
                study_id,
                _raw_volume(release_root, study_id),
                cache_dir,
                spec,
                overwrite=args.overwrite,
            )
            validation_errors = validate_cache_entry(result["preprocessed_path"], spec)
            if validation_errors:
                raise ValueError("; ".join(validation_errors))
            cache_rows[study_id] = result
        except Exception as exc:  # noqa: BLE001 - retain every failure in the audit
            failures.append({"study_id": study_id, "error": f"{type(exc).__name__}: {exc}"})

    output_audit: dict[str, Any] = {}
    dropped_rows: list[dict[str, Any]] = []
    for name, rows in loaded.items():
        kept: list[dict[str, Any]] = []
        for row in rows:
            study_id = str(row["study_id"])
            result = cache_rows.get(study_id)
            if result is None:
                dropped_rows.append({
                    "manifest": name,
                    "patient_id": row["patient_id"],
                    "study_id": study_id,
                    "split": row["split"],
                    "reason": "ctfm_preprocessing_failed",
                })
                continue
            copied = dict(row)
            copied["raw_image_path"] = str(_raw_volume(release_root, study_id))
            copied["image_path"] = str(result["preprocessed_path"])
            kept.append(copied)
        output_audit[name] = _validate_manifest(name, kept) if kept else {"rows": 0}
        source_fields = list(loaded[name][0])
        if "raw_image_path" not in source_fields:
            source_fields.append("raw_image_path")
        _write_csv(output_manifest_dir / MANIFESTS[name], kept, source_fields)

    _write_csv(
        output_root / "preprocessing_failures.csv",
        failures,
        ["study_id", "error"],
    )
    _write_csv(
        output_root / "dropped_rows.csv",
        dropped_rows,
        ["manifest", "patient_id", "study_id", "split", "reason"],
    )
    shape_counts: Counter[str] = Counter()
    spacing_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    crop_fractions: list[float] = []
    for result in cache_rows.values():
        metadata_path = Path(str(result["metadata_path"]))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shape_counts[json.dumps(metadata.get("output_shape"), separators=(",", ":"))] += 1
        spacing_counts[json.dumps(
            [round(float(value), 5) for value in metadata.get("output_spacing_mm", [])],
            separators=(",", ":"),
        )] += 1
        dtype_counts[str(metadata.get("dtype") or "unknown")] += 1
        crop_fraction = metadata.get("crop_fraction")
        if crop_fraction is not None:
            crop_fractions.append(float(crop_fraction))
    dropped_by_split = Counter(str(row.get("split") or "unknown") for row in dropped_rows)
    studies_attempted = len(study_ids)
    studies_cached = len(cache_rows)
    ct_qc = {
        "contract": {
            "target_shape": list(spec.target_shape),
            "resample_spacing_mm": list(spec.resample_spacing_mm or ()),
            "orientation": spec.orientation or "native",
            "clip_hu": [spec.clip_min_hu, spec.clip_max_hu],
            "normalization": spec.normalize,
            "dtype": spec.dtype,
            "spatial_strategy": spec.spatial_strategy,
        },
        "preprocessing_fingerprint": spec.fingerprint(),
        "studies_attempted": studies_attempted,
        "studies_cached": studies_cached,
        "studies_failed": len(failures),
        "cache_coverage": (
            float(studies_cached / studies_attempted) if studies_attempted else 0.0
        ),
        "dropped_rows": len(dropped_rows),
        "dropped_rows_by_split": dict(sorted(dropped_by_split.items())),
        "validated_entries": studies_cached,
        "output_shape_counts": dict(sorted(shape_counts.items())),
        "output_spacing_mm_counts": dict(sorted(spacing_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "crop_fraction": {
            "min": min(crop_fractions) if crop_fractions else None,
            "mean": sum(crop_fractions) / len(crop_fractions) if crop_fractions else None,
            "max": max(crop_fractions) if crop_fractions else None,
        },
    }
    payload = {
        "source_dataset_root": str(dataset_root),
        "source_manifest_dir": str(source_manifest_dir),
        "raw_release_root": str(release_root),
        "official_split_policy": "read existing manifests; never create or reassign splits",
        "input_audit": input_audit,
        "output_audit": output_audit,
        "preprocessing": spec.as_dict(),
        "preprocessing_fingerprint": spec.fingerprint(),
        "studies_attempted": studies_attempted,
        "studies_cached": studies_cached,
        "preprocessing_failures": len(failures),
        "dropped_rows": len(dropped_rows),
        "ct_qc": ct_qc,
        "label_contract": {
            "diagnosis": ["pe_present"],
            "prognosis": list(PROGNOSIS_TARGETS),
            "censoring": "empty event label + *_status retained; dataset masks censored targets",
        },
    }
    (output_root / "dataset.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
