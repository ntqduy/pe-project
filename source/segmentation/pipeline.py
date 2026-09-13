from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from source.engine.experiment import atomic_write_json
from source.imaging.nifti import binary_dice, load_nifti, mask_qc
from source.imaging.preview import write_overlay_previews

from .lungmask import LungMaskRunner
from .totalsegmentator import TASK_CLASSES, TotalSegmentatorRunner

ANATOMIES = (
    "lung",
    "heart",
    "strict_heart",
    "myocardium",
    "mediastinum",
    "central_pa",
    "lung_arteries",
    "lung_veins",
    "lung_vessels",
    "airways",
    "pa_tree",
    "hilar_vessels",
    "body",
    "body_wall",
    "rv",
    "lv",
    "ra",
    "la",
    "lung_lungmask",
)
# Preview paths and the patient/study output hierarchy are part of cached rows.
STATE_SCHEMA_VERSION = 3

ANATOMY_TASK = {
    "lung": "total",
    "heart": "total",
    "strict_heart": "derived",
    "myocardium": "heartchambers_highres",
    "mediastinum": "trunk_cavities",
    "central_pa": "heartchambers_highres",
    "lung_arteries": "lung_vessels",
    "lung_veins": "lung_vessels",
    "lung_vessels": "derived",
    "airways": "lung_vessels",
    "pa_tree": "derived",
    "hilar_vessels": "derived",
    "body": "body",
    "body_wall": "derived",
    "rv": "heartchambers_highres",
    "lv": "heartchambers_highres",
    "ra": "heartchambers_highres",
    "la": "heartchambers_highres",
    "lung_lungmask": "lungmask",
}


def _source_image(row: Mapping[str, Any], data_root: Path, image_column: str) -> Path:
    image = Path(str(row[image_column]))
    return image if image.is_absolute() else (data_root / image).resolve()


def _device(gpu_id: int | None) -> str:
    return "cpu" if gpu_id is None else f"gpu:{gpu_id}"


def _cached_rows(state_path: Path) -> list[dict[str, Any]] | None:
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        mask_path = row.get("mask_path")
        if mask_path and not Path(str(mask_path)).is_file():
            return None
    return [dict(row) for row in rows]


def _process_study(
    row: Mapping[str, Any],
    *,
    data_root: Path,
    image_column: str,
    run_dir: Path,
    backend: str,
    segmentation_config: Mapping[str, Any],
    gpu_id: int | None,
    preview: bool,
    progress: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    patient_id = str(row.get("patient_id") or "")
    study_id = str(row.get("study_id") or "")
    if not patient_id or not study_id:
        raise ValueError("segmentation rows require patient_id and study_id")
    state_path = run_dir / "state" / patient_id / f"{study_id}.json"
    cached = _cached_rows(state_path)
    if cached is not None:
        if progress:
            progress(f"segmentation study={study_id} patient={patient_id} status=cached")
        return cached
    image_path = _source_image(row, data_root, image_column)
    study_root = run_dir / "masks" / patient_id / study_id
    if progress:
        progress(
            f"segmentation study={study_id} patient={patient_id} "
            f"device={_device(gpu_id)} status=started"
        )
    if backend == "totalsegmentator":
        runner = TotalSegmentatorRunner(
            executable=str(segmentation_config.get("executable") or "TotalSegmentator"),
            repository=Path(str(segmentation_config["repository"])).resolve(),
            weights_directory=Path(str(segmentation_config["weights_directory"])).resolve(),
            fast=bool(segmentation_config.get("fast", False)),
            body_wall_thickness_mm=float(segmentation_config.get("body_wall_thickness_mm", 15.0)),
            hilar_proximity_mm=float(segmentation_config.get("hilar_proximity_mm", 12.0)),
            extra_arguments=tuple(segmentation_config.get("extra_arguments") or ()),
        )
        generated = runner.run(
            image_path,
            study_root,
            device=_device(gpu_id),
            log=progress,
        )
        lungmask_config = dict(segmentation_config.get("lungmask") or {})
        if lungmask_config.get("enabled", True):
            try:
                lungmask = LungMaskRunner(
                    executable=str(lungmask_config.get("executable") or "lungmask"),
                    checkpoint=Path(str(lungmask_config.get("checkpoint") or "")).resolve(),
                    model_name=str(lungmask_config.get("model_name") or "R231"),
                    force_cpu=bool(lungmask_config.get("force_cpu", False)),
                )
                path = study_root / "canonical" / "lung_lungmask.nii.gz"
                lungmask.run(image_path, path, gpu_id=gpu_id, log=progress)
                generated["masks"]["lung_lungmask"] = path
            except Exception as exc:  # noqa: BLE001 - independent QC model may be unavailable
                generated["errors"]["lungmask"] = f"{type(exc).__name__}: {exc}"
    else:
        raise ValueError(f"unknown segmentation backend: {backend}")

    limits = dict(segmentation_config.get("qc_volume_ml") or {})
    rows: list[dict[str, Any]] = []
    for anatomy in ANATOMIES:
        path = generated["masks"].get(anatomy)
        task = ANATOMY_TASK[anatomy]
        item_provenance = dict(generated.get("mask_provenance", {}).get(anatomy) or {})
        if path is None:
            task_error = generated["errors"].get(anatomy) or generated["errors"].get(task)
            rows.append(
                {
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "image_path": str(image_path),
                    "anatomy": anatomy,
                    "status": "UNAVAILABLE",
                    "reason": task_error or "source_mask_not_generated",
                    "mask_path": None,
                    "source_model": item_provenance.get("source_model", "TotalSegmentator"),
                    "source_task": json.dumps(item_provenance.get("source_task", task)),
                    "postprocessing": item_provenance.get("postprocessing"),
                    "parameters": json.dumps(item_provenance.get("parameters", {}), sort_keys=True),
                    "is_approximation": item_provenance.get("is_approximation"),
                    "fallback": item_provenance.get("fallback"),
                }
            )
            continue
        anatomy_limits = dict(limits.get(anatomy) or {})
        qc = mask_qc(
            path,
            image_path,
            minimum_volume_ml=anatomy_limits.get("minimum"),
            maximum_volume_ml=anatomy_limits.get("maximum"),
        )
        rows.append(
            {
                "patient_id": patient_id,
                "study_id": study_id,
                "image_path": str(image_path),
                "anatomy": anatomy,
                "status": qc["status"],
                "reason": qc["reason"],
                "mask_path": str(path),
                "source_model": (
                    "LungMask"
                    if anatomy == "lung_lungmask"
                    else item_provenance.get("source_model", "TotalSegmentator")
                ),
                "source_task": json.dumps(item_provenance.get("source_task", task)),
                "source_classes": json.dumps(item_provenance.get("source_classes", [])),
                "postprocessing": item_provenance.get("postprocessing", "binarize(value>0)"),
                "parameters": json.dumps(item_provenance.get("parameters", {}), sort_keys=True),
                "is_approximation": bool(item_provenance.get("is_approximation", False)),
                "fallback": item_provenance.get("fallback"),
                "voxel_count": qc.get("voxel_count"),
                "volume_ml": qc.get("volume_ml"),
                "component_count": qc.get("component_count"),
                "shape": json.dumps(qc.get("shape")),
                "spacing": json.dumps(qc.get("spacing")),
                "orientation": json.dumps(qc.get("orientation")),
                "affine": json.dumps(qc.get("affine")),
                "provenance": json.dumps(generated["provenance"], sort_keys=True),
            }
        )

    lookup = {item["anatomy"]: item for item in rows}
    first = lookup["lung"].get("mask_path")
    second = lookup["lung_lungmask"].get("mask_path")
    if first and second:
        dice = binary_dice(load_nifti(first)[0], load_nifti(second)[0])
        thresholds = dict(segmentation_config.get("lung_cross_model_qc") or {})
        accept = float(thresholds.get("accept_dice", 0.9))
        fail = float(thresholds.get("fail_below_dice", 0.7))
        cross_status = "PASS" if dice >= accept else "FAIL" if dice < fail else "SUSPICIOUS"
        for name in ("lung", "lung_lungmask"):
            lookup[name]["cross_model_dice"] = dice
            lookup[name]["cross_model_status"] = cross_status
            if cross_status != "PASS" and lookup[name]["status"] == "PASS":
                lookup[name]["status"] = cross_status
                lookup[name]["reason"] = (
                    f"{lookup[name]['reason']};cross_model_lung_dice_{cross_status.lower()}"
                )

    if preview:
        preview_anatomies = {
            str(value)
            for value in segmentation_config.get(
                "preview_anatomies", ("lung", "strict_heart", "pa_tree")
            )
        }
        preview_slices = int(segmentation_config.get("preview_slices_per_mask", 2))
        if preview_slices not in {1, 2, 3}:
            raise ValueError("segmentation.preview_slices_per_mask must be 1, 2, or 3")
        study_previews = run_dir / "previews" / patient_id / study_id / "segmentation"
        for item in rows:
            if item.get("mask_path") and item["anatomy"] in preview_anatomies:
                anatomy_name = str(item["anatomy"])
                preview_paths = write_overlay_previews(
                    image_path,
                    Path(str(item["mask_path"])),
                    study_previews / anatomy_name,
                    study_id=study_id,
                    patient_id=patient_id,
                    anatomy=anatomy_name,
                    source=str(item["source_model"]),
                    status=str(item["status"]),
                    maximum_slices=preview_slices,
                    voxel_count=int(item.get("voxel_count") or 0),
                    physical_volume_mm3=(
                        float(item["volume_ml"]) * 1000.0
                        if item.get("volume_ml") is not None else None
                    ),
                    overlay_color="orange",
                    window_width=float(segmentation_config.get("preview_window_width", 700.0)),
                    window_level=float(segmentation_config.get("preview_window_level", 100.0)),
                )
                item["preview_paths"] = json.dumps([str(path) for path in preview_paths])
    atomic_write_json(
        state_path,
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "patient_id": patient_id,
            "study_id": study_id,
            "rows": rows,
        },
    )
    if progress:
        available = sum(bool(item.get("mask_path")) for item in rows)
        progress(
            f"segmentation study={study_id} patient={patient_id} "
            f"status=completed masks={available}/{len(rows)}"
        )
    return rows


def generate_pseudo_anatomy(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    image_column: str,
    run_dir: Path,
    segmentation_config: Mapping[str, Any],
    maximum_cases: int | None,
    gpu_ids: Sequence[int],
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = list(source_rows if maximum_cases is None else source_rows[:maximum_cases])
    backend = str(segmentation_config.get("backend") or "totalsegmentator")
    raw_preview_cases = segmentation_config.get("preview_cases")
    preview_cases = None if raw_preview_cases is None else max(0, int(raw_preview_cases))
    assignments = list(gpu_ids) or [None]
    # One worker per GPU by default. segmentation.workers raises that independently, which
    # is what a CPU run needs: the device list is empty there, so the default would be a
    # single worker and the whole run would serialise. Studies keep round-robining over the
    # devices regardless of how many workers draw from them.
    configured_workers = segmentation_config.get("workers")
    workers = (
        len(assignments) if configured_workers is None else max(1, int(configured_workers))
    )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def submit(index: int, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        return _process_study(
            row,
            data_root=data_root,
            image_column=image_column,
            run_dir=run_dir,
            backend=backend,
            segmentation_config=segmentation_config,
            gpu_id=assignments[index % len(assignments)],
            preview=preview_cases is None or index < preview_cases,
            progress=progress,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(submit, index, row): row for index, row in enumerate(selected)}
        for future in as_completed(futures):
            source = futures[future]
            try:
                results.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve per-study failure accounting
                study_id = str(source.get("study_id") or "")
                message = (
                    f"segmentation study={study_id} "
                    f"patient={source.get('patient_id')} status=failed "
                    f"reason={type(exc).__name__}: {exc}"
                )
                if progress:
                    progress(message)
                failures.append(
                    {
                        "patient_id": str(source.get("patient_id") or ""),
                        "study_id": str(source.get("study_id") or ""),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    results.sort(key=lambda item: (str(item["study_id"]), str(item["anatomy"])))
    counts = Counter((str(item["anatomy"]), str(item["status"])) for item in results)
    cross = [
        float(item["cross_model_dice"])
        for item in results
        if item["anatomy"] == "lung" and item.get("cross_model_dice") is not None
    ]
    cross_status = Counter(
        str(item.get("cross_model_status"))
        for item in results
        if item["anatomy"] == "lung" and item.get("cross_model_status")
    )
    evaluation = {
        "backend": backend,
        "studies": {
            "requested": len(selected),
            "processed": len(selected) - len(failures),
            "failed": len(failures),
        },
        "failures": failures,
        "anatomy": {
            name: {status: counts[(name, status)] for status in ("PASS", "SUSPICIOUS", "FAIL", "UNAVAILABLE")}
            for name in ANATOMIES
        },
        "cross_model_lung_qc": {
            "both_available": len(cross),
            "pass": cross_status["PASS"],
            "suspicious": cross_status["SUSPICIOUS"],
            "fail": cross_status["FAIL"],
            "mean_dice": float(np.mean(cross)) if cross else None,
            "median_dice": float(np.median(cross)) if cross else None,
            "min_dice": float(np.min(cross)) if cross else None,
            "max_dice": float(np.max(cross)) if cross else None,
        },
        "totalsegmentator_tasks": {task: list(classes) for task, classes in TASK_CLASSES.items()},
    }
    return results, evaluation
