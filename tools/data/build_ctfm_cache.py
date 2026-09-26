#!/usr/bin/env python
"""Build the CT-FM feature cache from an already-built official-split dataset.

This stage never creates a split: it reads the project's filtered task manifests, verifies
their train/validation/test assignments, and writes CT-FM copies of them whose image_path
points at per-study CT-FM features.

Features follow the upstream CT-FM feature-extraction contract
(third_party/repos/CT-FM/scripts/feature_extractor.py):

    raw NIfTI -> SPL orientation -> 3 x 1 x 1 mm (trilinear) -> clip [-1024, 2048] HU
    -> [0, 1] -> body crop -> non-overlapping 24 x 128 x 128 patches -> SegResEncoder

The body-cropped volume is placed on a fixed canvas (default 120 x 384 x 384 voxels =
360 x 384 x 384 mm, a whole number of patches; larger bodies are centre-cropped, smaller
ones padded with air). Each study is stored as one float16 [513, d, h, w] array: the 512
deepest-level CT-FM channels stitched from all patches, plus one channel with the fraction
of each feature cell that lies inside the body box, so padding never enters the pooled
embedding. The sidecar records the preprocessing spec, the canvas geometry and the feature
grid affine used to put ROI masks on the same grid.

Each grid is also pooled once into pooled/<study_id>.npy, float32 [513, 1, 1, 1]: the
body-weighted mean the frozen-CT-FM MLP computes from the grid (ct_fm_cached_feature_adapter)
plus a weight channel of 1, so the unchanged model reads it as a one-cell grid and gets the
same global embedding. Training on these ~2 KB files instead of the ~5.9 MB grids removes
the per-epoch re-reading of every grid; previews and the anatomy arms still use the grid.

``--workers auto`` (default) sizes the preprocessing pool from the CPUs and the RAM free at
start (source/utils/workers.py); an explicit number is capped when memory cannot hold it.

Outputs (profile = dataset directory name):
    <derived>/cache/<profile>/ct_fm/features/<study_id>.npy (+ .metadata.json)
    <derived>/cache/<profile>/ct_fm/pooled/<study_id>.npy
    <derived>/cache/<profile>/ct_fm/dataset.json, preprocessing_failures.csv, dropped_rows.csv
    <derived>/datasets/<profile>/manifests/ct_fm/*.csv   (+ README.txt)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from source.data.manifests import read_rows
from source.data_preprocessing.volumes import PreprocessingSpec, preprocess_volume_with_metadata
from source.utils.progress import format_duration
from source.utils.workers import PREPROCESS_WORKER_GB, resolve_workers

DEFAULT_OUTPUT_NAME = "ct_fm"
DEFAULT_CANVAS = (120, 384, 384)
FEATURE_IMPLEMENTATION = "ctfm-features-v1"
CTFM_WEIGHT = Path("third_party/weights/ct_fm_feature_extractor/model.safetensors")
CTFM_WEIGHT_SHA256 = "b521ff13764ad0fce67f8ad2e5aa9ccc0823f69e4544b446581d8a2dee686215"
MANIFEST_README = """CT-FM task manifests (written by tools/data/build_ctfm_cache.py)

These are the base task manifests in ../ (diagnosis, prognosis, prognosis_all_patient,
prognosis_pe_positive) with two differences only:
  * image_path points at the study's CT-FM feature file
    (<derived>/cache/<profile>/ct_fm/features/<study_id>.npy, float16 [513, d, h, w]);
  * pooled_path points at its pooled copy
    (<derived>/cache/<profile>/ct_fm/pooled/<study_id>.npy, float32 [513, 1, 1, 1]), which
    the frozen CT-FM + MLP runs read (data.file_column: pooled_path);
  * studies whose CT-FM preprocessing failed are dropped (listed in
    <derived>/cache/<profile>/ct_fm/dropped_rows.csv).
Splits, labels and every other column are copied unchanged. Use them only with
model.backbone=ct_fm_features (configs/runs/01_foundation/ct_fm_frozen_*.yaml); every other
model reads the base manifests in ../.
"""


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


def update_ctfm_quality_report(dataset_root: Path, payload: dict[str, Any]) -> None:
    """Add CT-FM cache counts to the dataset's single human-readable QC report."""
    report_path = dataset_root / "data_quality.md"
    if not report_path.is_file():
        return
    content = report_path.read_text(encoding="utf-8")
    marker = "\n## CT-FM cache QC\n"
    start = content.find(marker)
    if start >= 0:
        next_section = content.find("\n## ", start + len(marker))
        content = content[:start] + (content[next_section:] if next_section >= 0 else "")
    qc = dict(payload.get("ct_qc") or {})
    split_counts = dict(qc.get("dropped_rows_by_split") or {})
    lines = [
        "## CT-FM cache QC",
        "",
        f"- Studies cached: {payload['studies_cached']}/{payload['studies_attempted']}.",
        f"- Preprocessing failures: {payload['preprocessing_failures']}.",
        f"- Manifest rows dropped: {payload['dropped_rows']}; by split: "
        + (", ".join(f"{key}={value}" for key, value in sorted(split_counts.items())) or "none")
        + ".",
        "- Output tensor: " + str(qc.get("output_shape_counts") or {})
        + " (512 CT-FM feature channels + 1 body-coverage channel, float16)",
        "- Contract: SPL, 3x1x1 mm, [-1024, 2048] HU -> [0, 1], 24x128x128 patches; canvas "
        + str((qc.get("contract") or {}).get("canvas_voxels"))
        + f"; studies centre-cropped to the canvas: {qc.get('studies_centre_cropped_to_canvas')}.",
    ]
    if payload["preprocessing_failures"] or payload["dropped_rows"]:
        lines.append("Affected IDs are in external CT-FM cache: preprocessing_failures.csv and dropped_rows.csv.")
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(content.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report_path)


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


def _spec(canvas: tuple[int, int, int]) -> PreprocessingSpec:
    return PreprocessingSpec(
        target_shape=canvas,
        clip_min_hu=-1024.0,
        clip_max_hu=2048.0,
        normalize="minmax",
        dtype="float32",
        orientation="SPL",
        resample_spacing_mm=(3.0, 1.0, 1.0),
        foreground_strategy="external_body",
        foreground_threshold_hu=0.0,
        foreground_margin_mm=0.0,
        spatial_strategy="centre_crop_or_pad",
        output_format="npy",
        emit_patch_grid=False,
        extra={
            "foundation_model": "CT-FM",
            "upstream_contract": "SPL; spacing=[3,1,1]; CropForeground; ScaleIntensityRanged[-1024,2048]; "
            "SlidingWindowSplitter(24x128x128, overlap 0)",
            "mode": "patch_features",
        },
    )


def _preprocess(job: tuple[str, str, dict[str, Any]]) -> tuple[str, Any, Any, str | None]:
    """CPU part, run in worker processes: raw NIfTI -> canvas volume + geometry metadata."""
    study_id, raw_path, spec_payload = job
    try:
        spec = PreprocessingSpec.from_mapping(spec_payload)
        volume, metadata = preprocess_volume_with_metadata(raw_path, spec)
        return study_id, np.asarray(volume, dtype=np.float32), metadata, None
    except Exception as exc:  # noqa: BLE001 - every failure is recorded per study
        return study_id, None, None, f"{type(exc).__name__}: {exc}"


def _bounded_map(executor: ProcessPoolExecutor, function: Any, jobs: list[Any], limit: int):
    """``executor.map`` in submission order with at most ``limit`` results in flight, so
    preprocessed volumes (~70 MB each) never pile up in memory behind the GPU."""
    from collections import deque

    pending: deque = deque()
    iterator = iter(jobs)
    for job in iterator:
        pending.append(executor.submit(function, job))
        if len(pending) >= limit:
            break
    while pending:
        yield pending.popleft().result()
        for job in iterator:
            pending.append(executor.submit(function, job))
            break


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ctfm(code_root: Path, device: Any) -> Any:
    """The public CT-FM SegResEncoder, checksum-verified and strict-loaded."""
    import torch
    from safetensors.torch import load_file

    from source.components.encoders.image.ct_fm import build_ct_fm_backbone

    weight = code_root / CTFM_WEIGHT
    if not weight.is_file():
        raise SystemExit(f"CT-FM weight not found: {weight}")
    checksum = _sha256(weight)
    if checksum != CTFM_WEIGHT_SHA256:
        raise SystemExit(f"CT-FM weight checksum mismatch: {checksum} != {CTFM_WEIGHT_SHA256}")
    model = build_ct_fm_backbone()
    model.load_state_dict(load_file(str(weight), device="cpu"), strict=True)
    model.eval().to(device)
    torch.set_grad_enabled(False)
    return model, checksum


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npy")
    try:
        np.save(temporary, array)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pooled_from_grid(array: np.ndarray) -> np.ndarray:
    """float32 [513, 1, 1, 1]: the body-weighted mean the frozen MLP pools from ``array``.

    Same formula as ct_fm_cached_feature_adapter (weights = coverage channel clamped to
    [0, 1], denominator clamped at 1e-6), accumulated in float64 from the stored float16
    values. The last channel is 1, so the adapter's mean over this one cell returns the
    512 pooled values unchanged.
    """
    values = np.asarray(array, dtype=np.float64)
    weights = np.clip(values[-1], 0.0, 1.0)
    pooled = (values[:-1] * weights).sum(axis=(1, 2, 3)) / max(float(weights.sum()), 1e-6)
    return np.concatenate([pooled, [1.0]]).astype(np.float32).reshape(-1, 1, 1, 1)


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def _pooled_current(feature_path: Path, pooled_path: Path) -> bool:
    return (pooled_path.is_file() and _sidecar(pooled_path).is_file()
            and pooled_path.stat().st_mtime >= feature_path.stat().st_mtime)


def _ensure_pooled(
    feature_path: Path,
    pooled_path: Path,
    array: np.ndarray | None = None,
    sidecar: dict[str, Any] | None = None,
) -> None:
    """Write the pooled copy and its sidecar, from ``array`` or the grid on disk when stale.

    The sidecar copies the grid's (representation, weight hash, fingerprint and source are what
    the dataset's input-contract check reads) minus the feature-grid geometry, which a single
    pooled cell does not have.
    """
    if array is None:
        if _pooled_current(feature_path, pooled_path):
            return
        array = np.load(feature_path)
    if sidecar is None:
        sidecar = json.loads(_sidecar(feature_path).read_text(encoding="utf-8"))
    pooled = _pooled_from_grid(array)
    _atomic_save(pooled_path, pooled)
    _atomic_json(_sidecar(pooled_path), {
        **{key: value for key, value in sidecar.items() if key not in {"feature_grid", "coverage_channel"}},
        "tensor_shape": list(pooled.shape),
        "tensor_dtype": str(pooled.dtype),
        "pooled_from": str(feature_path),
        "pooling": "body-weighted mean over feature cells (coverage channel clamped to [0, 1]); "
                   "channel 512 is the weight 1 of the single cell",
    })


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _feature_grid_affine(canvas_affine: np.ndarray, cell: tuple[int, int, int]) -> np.ndarray:
    """Voxel->world affine of the feature grid: cell (i, j, k) covers canvas voxels
    [i*c0, (i+1)*c0) x ...; its centre is canvas voxel (i + 0.5) * c - 0.5."""
    transform = np.eye(4)
    transform[:3, :3] = np.diag(np.asarray(cell, dtype=float))
    transform[:3, 3] = (np.asarray(cell, dtype=float) - 1.0) / 2.0
    return np.asarray(canvas_affine, dtype=float) @ transform


def _cached_entry(feature_path: Path, fingerprint: str) -> dict[str, Any] | None:
    sidecar = feature_path.with_name(feature_path.name + ".metadata.json")
    if not (feature_path.is_file() and sidecar.is_file()):
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "completed" or payload.get("cache_fingerprint") != fingerprint:
        return None
    if payload.get("representation") != "ct_fm_features_v1" or payload.get("weight_sha256") != CTFM_WEIGHT_SHA256:
        return None
    try:
        array = np.load(feature_path, mmap_mode="r")
        if (array.ndim != 4 or array.shape[0] != 513 or array.dtype != np.float16
                or list(array.shape) != payload.get("tensor_shape")):
            return None
    except (OSError, ValueError):
        return None
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="derived/datasets/<profile> containing the existing manifests")
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="INSPECT root or directory containing CT/full/CTPA")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--cache-root", type=Path,
                        help="external profile cache root (default: <derived>/cache/<profile>)")
    parser.add_argument("--canvas", default=",".join(str(v) for v in DEFAULT_CANVAS),
                        help="S,P,L canvas in 3x1x1 mm voxels; multiples of 24,128,128")
    parser.add_argument("--device", default=None, help="torch device (default: cuda when available)")
    parser.add_argument("--patch-batch", type=int, default=16, help="CT-FM patches per forward pass")
    parser.add_argument("--workers", default="auto",
                        help="processes for NIfTI loading/resampling: auto (CPUs and free RAM) or a "
                             "number, capped when the free RAM cannot hold it")
    parser.add_argument("--overwrite", action="store_true", help="recompute every study")
    parser.add_argument("--quiet", action="store_true",
                        help="print a compact summary; the full QC remains in dataset.json")
    args = parser.parse_args()

    import torch

    from source.components.encoders.image.ct_fm import (
        CTFM_FEATURE_REPRESENTATION,
        CTFM_PATCH_SIZE,
        ctfm_patch_features,
    )
    from source.imaging.grid import cell_fraction

    code_root = Path(__file__).resolve().parents[2]
    dataset_root = args.dataset_root.resolve()
    release_root = _release_root(args.raw_root.resolve())
    source_manifest_dir = dataset_root / "manifests"
    if not args.output_name or Path(args.output_name).name != args.output_name or args.output_name in {".", ".."}:
        parser.error("--output-name must be one directory name")
    canvas = tuple(int(value) for value in str(args.canvas).split(","))
    if len(canvas) != 3 or any(size % step for size, step in zip(canvas, CTFM_PATCH_SIZE)):
        parser.error(f"--canvas must be three multiples of {CTFM_PATCH_SIZE}")
    profile_cache_root = args.cache_root.resolve() if args.cache_root else (
        dataset_root.parent.parent / "cache" / dataset_root.name
    )
    output_root = profile_cache_root / args.output_name
    output_manifest_dir = source_manifest_dir / args.output_name
    feature_dir = output_root / "features"
    pooled_dir = output_root / "pooled"
    output_root.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, list[dict[str, Any]]] = {}
    input_audit: dict[str, Any] = {}
    for name, filename in MANIFESTS.items():
        rows = read_rows(source_manifest_dir / filename)
        input_audit[name] = _validate_manifest(name, rows)
        _validate_labels(name, rows)
        loaded[name] = rows

    spec = _spec(canvas)  # type: ignore[arg-type]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, weight_sha256 = _load_ctfm(code_root, device)
    fingerprint = hashlib.sha256(
        json.dumps(
            {"spec": spec.fingerprint(), "implementation": FEATURE_IMPLEMENTATION, "weight": weight_sha256},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    study_ids = sorted({str(row["study_id"]) for rows in loaded.values() for row in rows})
    cache_rows: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    jobs: list[tuple[str, str, dict[str, Any]]] = []
    for study_id in study_ids:
        feature_path = feature_dir / f"{study_id}.npy"
        cached = None if args.overwrite else _cached_entry(feature_path, fingerprint)
        if cached is not None:
            cache_rows[study_id] = {"path": feature_path, "metadata": cached}
            continue
        try:
            jobs.append((study_id, str(_raw_volume(release_root, study_id)), spec.as_dict()))
        except FileNotFoundError as exc:
            failures.append({"study_id": study_id, "error": f"FileNotFoundError: {exc}"})
    started = time.perf_counter()
    # Sized after the model is loaded, so the free RAM already excludes the main process.
    workers, worker_note = resolve_workers(
        args.workers, per_worker_gb=PREPROCESS_WORKER_GB, cpu_margin=1, minimum=1
    )
    print(
        f"CT-FM features: {len(study_ids)} studies, {len(cache_rows)} cached, {len(jobs)} to compute "
        f"on {device} (canvas {canvas}, {worker_note})",
        flush=True,
    )
    raw_paths = {job[0]: job[1] for job in jobs}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = _bounded_map(executor, _preprocess, jobs, 2 * workers)
        for done, (study_id, volume, metadata, error) in enumerate(results, start=1):
            if error is not None:
                failures.append({"study_id": study_id, "error": error})
                continue
            try:
                tensor = torch.from_numpy(volume)[None, None].to(device)
                features = ctfm_patch_features(model, tensor, batch_size=int(args.patch_batch))[0]
                grid_shape = tuple(int(value) for value in features.shape[-3:])
                cell = tuple(size // cells for size, cells in zip(canvas, grid_shape))
                valid = np.zeros(canvas, dtype=np.float32)
                bounds = metadata["target_bounds"]
                valid[tuple(slice(int(low), int(high)) for low, high in bounds)] = 1.0
                coverage = cell_fraction(valid, cell)
                array = np.concatenate(
                    [features.float().cpu().numpy(), coverage[None]], axis=0
                ).astype(np.float16)
                if not np.isfinite(array).all():
                    raise ValueError("non-finite CT-FM features")
                feature_path = feature_dir / f"{study_id}.npy"
                _atomic_save(feature_path, array)
                sidecar = {
                    **{key: value for key, value in metadata.items()},
                    "status": "completed",
                    "representation": CTFM_FEATURE_REPRESENTATION,
                    "feature_implementation": FEATURE_IMPLEMENTATION,
                    "cache_fingerprint": fingerprint,
                    "study_id": study_id,
                    "raw_image_path": raw_paths[study_id],
                    "spec": spec.as_dict(),
                    "preprocessing_fingerprint": spec.fingerprint(),
                    "weight_sha256": weight_sha256,
                    "patch_size": list(CTFM_PATCH_SIZE),
                    "tensor_shape": list(array.shape),
                    "tensor_dtype": str(array.dtype),
                    "feature_channels": int(features.shape[0]),
                    "coverage_channel": int(features.shape[0]),
                    "feature_grid": {
                        "shape": list(grid_shape),
                        "cell_voxels": list(cell),
                        "affine": _feature_grid_affine(np.asarray(metadata["output_affine"]), cell).tolist(),
                    },
                    "body_cell_fraction": float(coverage.mean()),
                    "canvas_cropped": bool(
                        any(
                            int(source[1]) - int(source[0]) < int(full)
                            for source, full in zip(metadata["source_bounds"], metadata["cropped_shape_before_fit"])
                        )
                    ),
                }
                _atomic_json(feature_path.with_name(feature_path.name + ".metadata.json"), sidecar)
                _ensure_pooled(feature_path, pooled_dir / f"{study_id}.npy", array, sidecar)
                cache_rows[study_id] = {"path": feature_path, "metadata": sidecar}
            except Exception as exc:  # noqa: BLE001 - retain every failure in the audit
                failures.append({"study_id": study_id, "error": f"{type(exc).__name__}: {exc}"})
            if not args.quiet or done % 25 == 0 or done == len(jobs):
                elapsed = time.perf_counter() - started
                left = format_duration(elapsed / done * (len(jobs) - done))
                print(f"  {done}/{len(jobs)} computed ({elapsed / done:.1f} s/study, ~{left} left, "
                      f"{len(failures)} failed)", flush=True)

    # Grids cached by an earlier run (or before pooling existed) get their pooled copy here;
    # reading 5.9 MB files over gcsfuse is latency-bound, hence threads.
    backfill = [study_id for study_id in cache_rows
                if not _pooled_current(cache_rows[study_id]["path"], pooled_dir / f"{study_id}.npy")]
    if backfill:
        pooled_started = time.perf_counter()
        print(f"CT-FM pooled copies: {len(backfill)} to write from cached grids", flush=True)
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(_ensure_pooled, cache_rows[study_id]["path"],
                                   pooled_dir / f"{study_id}.npy"): study_id for study_id in backfill}
            for done, future in enumerate(as_completed(futures), start=1):
                future.result()
                if done % 500 == 0 or done == len(backfill):
                    elapsed = time.perf_counter() - pooled_started
                    left = format_duration(elapsed / done * (len(backfill) - done))
                    print(f"  {done}/{len(backfill)} pooled (~{left} left)", flush=True)

    output_audit: dict[str, Any] = {}
    dropped_rows: list[dict[str, Any]] = []
    for name, rows in loaded.items():
        kept: list[dict[str, Any]] = []
        for row in rows:
            study_id = str(row["study_id"])
            entry = cache_rows.get(study_id)
            if entry is None:
                dropped_rows.append({
                    "manifest": name,
                    "patient_id": row["patient_id"],
                    "study_id": study_id,
                    "split": row["split"],
                    "reason": "ctfm_feature_extraction_failed",
                })
                continue
            copied = dict(row)
            copied["raw_image_path"] = str(entry["metadata"].get("raw_image_path") or row.get("raw_image_path") or "")
            copied["image_path"] = str(entry["path"])
            copied["pooled_path"] = str(pooled_dir / f"{study_id}.npy")
            kept.append(copied)
        output_audit[name] = _validate_manifest(name, kept) if kept else {"rows": 0}
        source_fields = list(loaded[name][0])
        for field in ("raw_image_path", "pooled_path"):
            if field not in source_fields:
                source_fields.append(field)
        _write_csv(output_manifest_dir / MANIFESTS[name], kept, source_fields)
    (output_manifest_dir / "README.txt").write_text(MANIFEST_README, encoding="utf-8")

    for filename, rows, fields in (
        ("preprocessing_failures.csv", failures, ["study_id", "error"]),
        ("dropped_rows.csv", dropped_rows, ["manifest", "patient_id", "study_id", "split", "reason"]),
    ):
        path = output_root / filename
        if rows:
            _write_csv(path, rows, fields)
        else:
            path.unlink(missing_ok=True)
    metadata_rows = [entry["metadata"] for entry in cache_rows.values()]
    shape_counts = Counter(json.dumps(item.get("tensor_shape"), separators=(",", ":")) for item in metadata_rows)
    cropped = sum(bool(item.get("canvas_cropped")) for item in metadata_rows)
    coverage = [float(item.get("body_cell_fraction") or 0.0) for item in metadata_rows]
    dropped_by_split = Counter(str(row.get("split") or "unknown") for row in dropped_rows)
    studies_attempted = len(study_ids)
    studies_cached = len(cache_rows)
    ct_qc = {
        "contract": {
            "representation": "ct_fm_features_v1",
            "canvas_voxels": list(canvas),
            "resample_spacing_mm": list(spec.resample_spacing_mm or ()),
            "orientation": spec.orientation or "native",
            "clip_hu": [spec.clip_min_hu, spec.clip_max_hu],
            "normalization": spec.normalize,
            "patch_size": list(CTFM_PATCH_SIZE),
            "tensor": "float16 [512 features + 1 body coverage, d, h, w]",
            "pooled_tensor": "float32 [512 body-weighted mean features + 1 weight (=1), 1, 1, 1] in pooled/",
        },
        "cache_fingerprint": fingerprint,
        "weight_sha256": weight_sha256,
        "studies_attempted": studies_attempted,
        "studies_cached": studies_cached,
        "studies_failed": len(failures),
        "cache_coverage": float(studies_cached / studies_attempted) if studies_attempted else 0.0,
        "dropped_rows": len(dropped_rows),
        "dropped_rows_by_split": dict(sorted(dropped_by_split.items())),
        "output_shape_counts": dict(sorted(shape_counts.items())),
        "dtype_counts": {"float16": studies_cached},
        "studies_centre_cropped_to_canvas": cropped,
        "body_cell_fraction": {
            "min": min(coverage) if coverage else None,
            "mean": sum(coverage) / len(coverage) if coverage else None,
            "max": max(coverage) if coverage else None,
        },
    }
    payload = {
        "source_dataset_root": str(dataset_root),
        "source_manifest_dir": str(source_manifest_dir),
        "output_manifest_dir": str(output_manifest_dir),
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
    _atomic_json(output_root / "dataset.json", payload)
    update_ctfm_quality_report(dataset_root, payload)
    summary = (
        f"CT-FM feature cache ready: studies={studies_cached}/{studies_attempted} "
        f"failures={len(failures)} dropped_rows={len(dropped_rows)} canvas={canvas} "
        f"centre_cropped={cropped} output={output_root} manifests={output_manifest_dir}"
    )
    print(summary if args.quiet else json.dumps(payload, indent=2, sort_keys=True) + "\n" + summary)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
