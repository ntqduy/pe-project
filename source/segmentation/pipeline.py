from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from source.engine.experiment import atomic_write_json
from source.imaging.nifti import binary_dice, load_nifti, mask_qc
from source.imaging.preview import write_segmentation_contact_sheet

from .labels import parse_pe_present
from .lungmask import LungMaskRunner
from .resume import fingerprint_matches, guard_resume_settings, without_keys
from .totalsegmentator import LUNG_LOBES, LUNG_SIDES, TASK_CLASSES, TotalSegmentatorRunner

ANATOMIES = (
    "lung",
    *LUNG_SIDES,
    *LUNG_LOBES,
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
# 4: flat masks/<patient>/<study>/<anatomy>.nii.gz, one contact-sheet preview per study.
# 5: + lung_left/lung_right and the five lung lobes (26 masks per study).
STATE_SCHEMA_VERSION = 5
# Segmentation settings that only change speed, devices, paths or diagnostic previews. Every
# other key can change the masks, so a resumed run must have been built with the same values
# (see source/segmentation/resume.py).
RESUME_IGNORED_KEYS = frozenset({
    "devices",
    "executable",
    "preview_cases",
    "preview_window_level",
    "preview_window_width",
    "raw_image_root",
    "repository",
    "scratch_dir",
    "task_timeout_sec",
    "totalseg_resample_threads",
    "totalseg_saving_threads",
    "weights_directory",
    "workers",
})
LUNGMASK_RESUME_IGNORED_KEYS = frozenset({"checkpoint", "executable", "force_cpu"})

ANATOMY_TASK = {
    "lung": "total",
    **{name: "total" for name in (*LUNG_SIDES, *LUNG_LOBES)},
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


def _source_image(
    row: Mapping[str, Any],
    data_root: Path,
    image_column: str,
    raw_image_root: Path | None = None,
) -> Path:
    image = Path(str(row[image_column]))
    resolved = image if image.is_absolute() else (data_root / image).resolve()
    if resolved.suffix == ".npy" and raw_image_root is not None:
        # Preprocessing manifests point at dense .npy caches, while the external
        # segmentation tools require the original NIfTI geometry. Keep the cache
        # path in the manifest and resolve the raw study only for this stage.
        study_id = str(row.get("study_id") or "").strip()
        if study_id:
            raw = raw_image_root / f"{study_id}.nii.gz"
            if raw.is_file():
                return raw.resolve()
    return resolved.resolve()


def _device(gpu_id: int | None) -> str:
    return "cpu" if gpu_id is None else f"gpu:{gpu_id}"


def resume_settings(segmentation_config: Mapping[str, Any]) -> dict[str, Any]:
    """Output-affecting segmentation settings; a resumed run must match them exactly."""
    settings = without_keys(segmentation_config, RESUME_IGNORED_KEYS)
    # The defaults _process_study applies, so an omitted key equals its explicit default.
    settings.update(
        backend=str(settings.get("backend") or "totalsegmentator"),
        fast=bool(settings.get("fast", False)),
        body_wall_thickness_mm=float(settings.get("body_wall_thickness_mm", 15.0)),
        hilar_proximity_mm=float(settings.get("hilar_proximity_mm", 12.0)),
        extra_arguments=[str(value) for value in settings.get("extra_arguments") or ()],
    )
    settings["lungmask"] = without_keys(
        segmentation_config.get("lungmask"), LUNGMASK_RESUME_IGNORED_KEYS
    )
    return settings


def resume_settings_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return resume_settings(dict(snapshot.get("segmentation") or {}))


def _cached_rows(state_path: Path, settings_fingerprint: str | None = None) -> list[dict[str, Any]] | None:
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    # States written before fingerprints existed are covered by the run-level check.
    recorded = payload.get("settings_fingerprint")
    if (
        settings_fingerprint is not None
        and recorded is not None
        and not fingerprint_matches(recorded, settings_fingerprint)
    ):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(ANATOMIES):
        return None
    if {str(row.get("anatomy")) for row in rows if isinstance(row, dict)} != set(ANATOMIES):
        return None
    for row in rows:
        mask_path = row.get("mask_path")
        if mask_path and not Path(str(mask_path)).is_file():
            return None
        # A missing canonical mask is a retryable task failure, not completed state.
        if row.get("anatomy") != "lung_lungmask" and row.get("status") == "UNAVAILABLE":
            return None
    return [dict(row) for row in rows]


def study_preview_path(run_dir: Path, patient_id: str, study_id: str) -> Path:
    """One contact-sheet PNG per study, directly under ``previews/``."""
    return run_dir / "previews" / f"{patient_id}_{study_id}.png"


def _remove_legacy_previews(run_dir: Path, patient_id: str, study_id: str) -> None:
    """Drop the old ``previews/<patient>/<study>/segmentation/<anatomy>/*.png`` tree."""
    legacy = run_dir / "previews" / patient_id / study_id
    if legacy.is_dir():
        shutil.rmtree(legacy, ignore_errors=True)
        parent = legacy.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def _assign_duplicates(rows: list[dict[str, Any]]) -> None:
    """Flag voxel-identical masks, e.g. pa_tree == lung_arteries when the latter holds the PA."""
    groups: dict[str, list[str]] = {}
    for item in rows:
        digest = item.get("mask_sha1")
        if item.get("mask_path") and digest and int(item.get("voxel_count") or 0) > 0:
            groups.setdefault(str(digest), []).append(str(item["anatomy"]))
    for item in rows:
        members = groups.get(str(item.get("mask_sha1") or ""), [])
        others = [name for name in members if name != item["anatomy"]]
        item["duplicate_of"] = ";".join(others)


def _write_study_preview(
    rows: list[dict[str, Any]],
    *,
    image_path: Path,
    run_dir: Path,
    patient_id: str,
    study_id: str,
    segmentation_config: Mapping[str, Any],
    note: str,
    pe_present: bool | None,
    progress: Callable[[str], None] | None,
) -> str:
    """Write the study contact sheet; a preview failure never fails the study."""
    destination = study_preview_path(run_dir, patient_id, study_id)
    try:
        write_segmentation_contact_sheet(
            image_path,
            rows,
            destination,
            study_id=study_id,
            patient_id=patient_id,
            window_width=float(segmentation_config.get("preview_window_width", 700.0)),
            window_level=float(segmentation_config.get("preview_window_level", 100.0)),
            title_note=note,
            pe_present=pe_present,
        )
    except Exception as exc:  # noqa: BLE001 - preview is diagnostic output only
        if progress:
            progress(
                f"segmentation study={study_id} patient={patient_id} "
                f"preview=failed reason={type(exc).__name__}: {exc}"
            )
        return ""
    _remove_legacy_previews(run_dir, patient_id, study_id)
    if progress:
        progress(f"segmentation study={study_id} patient={patient_id} preview={destination}")
    return str(destination)


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
    settings_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    patient_id = str(row.get("patient_id") or "")
    study_id = str(row.get("study_id") or "")
    pe_present = parse_pe_present(row.get("pe_present"))
    if not patient_id or not study_id:
        raise ValueError("segmentation rows require patient_id and study_id")
    state_path = run_dir / "state" / patient_id / f"{study_id}.json"
    cached = _cached_rows(state_path, settings_fingerprint)
    if cached is not None:
        if progress:
            progress(f"segmentation study={study_id} patient={patient_id} status=cached")
        return cached
    raw_image_root_value = segmentation_config.get("raw_image_root")
    raw_image_root = Path(str(raw_image_root_value)).resolve() if raw_image_root_value else None
    image_path = _source_image(row, data_root, image_column, raw_image_root)
    # Flat layout: masks/<patient>/<study>/<anatomy>.nii.gz (one per ANATOMIES entry, 26).
    study_root = run_dir / "masks" / patient_id / study_id
    # A retry after a failed task must not leave an old mask beside the new output.
    for anatomy in ANATOMIES:
        (study_root / f"{anatomy}.nii.gz").unlink(missing_ok=True)
    if progress:
        progress(
            f"segmentation study={study_id} patient={patient_id} "
            f"device={_device(gpu_id)} status=started"
        )
    if backend != "totalsegmentator":
        raise ValueError(f"unknown segmentation backend: {backend}")
    scratch_value = segmentation_config.get("scratch_dir")
    runner = TotalSegmentatorRunner(
        executable=str(segmentation_config.get("executable") or "TotalSegmentator"),
        repository=Path(str(segmentation_config["repository"])).resolve(),
        weights_directory=Path(str(segmentation_config["weights_directory"])).resolve(),
        fast=bool(segmentation_config.get("fast", False)),
        body_wall_thickness_mm=float(segmentation_config.get("body_wall_thickness_mm", 15.0)),
        hilar_proximity_mm=float(segmentation_config.get("hilar_proximity_mm", 12.0)),
        extra_arguments=tuple(segmentation_config.get("extra_arguments") or ()),
        scratch_directory=Path(str(scratch_value)) if scratch_value else None,
        resample_threads=int(segmentation_config.get("totalseg_resample_threads", 1)),
        saving_threads=int(segmentation_config.get("totalseg_saving_threads", 1)),
        task_timeout_sec=(
            float(segmentation_config["task_timeout_sec"])
            if segmentation_config.get("task_timeout_sec") is not None
            else 3600.0
        ),
    )
    generated = runner.run(
        image_path,
        study_root,
        device=_device(gpu_id),
        log=progress,
    )

    limits = dict(segmentation_config.get("qc_volume_ml") or {})
    qc_cache: dict[str, dict[str, Any]] = {}

    def build_rows(pending: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for anatomy in ANATOMIES:
            path = generated["masks"].get(anatomy)
            task = ANATOMY_TASK[anatomy]
            item_provenance = dict(generated.get("mask_provenance", {}).get(anatomy) or {})
            if path is None:
                task_error = (
                    (pending or {}).get(anatomy)
                    or generated["errors"].get(anatomy)
                    or generated["errors"].get(task)
                )
                rows.append(
                    {
                        "patient_id": patient_id,
                        "study_id": study_id,
                        "split": row.get("split"),
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
            if anatomy not in qc_cache:
                anatomy_limits = dict(limits.get(anatomy) or {})
                qc_cache[anatomy] = mask_qc(
                    path,
                    image_path,
                    minimum_volume_ml=anatomy_limits.get("minimum"),
                    maximum_volume_ml=anatomy_limits.get("maximum"),
                )
            qc = qc_cache[anatomy]
            rows.append(
                {
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "split": row.get("split"),
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
                    "z_first": qc.get("z_first"),
                    "z_last": qc.get("z_last"),
                    "mask_sha1": qc.get("mask_sha1"),
                    "shape": json.dumps(qc.get("shape")),
                    "spacing": json.dumps(qc.get("spacing")),
                    "orientation": json.dumps(qc.get("orientation")),
                    "affine": json.dumps(qc.get("affine")),
                    "provenance": json.dumps(generated["provenance"], sort_keys=True),
                }
            )
        _assign_duplicates(rows)
        return rows

    lungmask_config = dict(segmentation_config.get("lungmask") or {})
    lungmask_enabled = bool(lungmask_config.get("enabled", True))
    if preview:
        # Written before LungMask so an interrupted study still leaves a reviewable image.
        early_rows = build_rows(
            {"lung_lungmask": "pending: LungMask not run yet"} if lungmask_enabled else None
        )
        _write_study_preview(
            early_rows,
            image_path=image_path,
            run_dir=run_dir,
            patient_id=patient_id,
            study_id=study_id,
            segmentation_config=segmentation_config,
            note="LungMask pending" if lungmask_enabled else "",
            pe_present=pe_present,
            progress=progress,
        )
    if lungmask_enabled:
        try:
            lungmask = LungMaskRunner(
                executable=str(lungmask_config.get("executable") or "lungmask"),
                checkpoint=Path(str(lungmask_config.get("checkpoint") or "")).resolve(),
                model_name=str(lungmask_config.get("model_name") or "R231"),
                force_cpu=bool(lungmask_config.get("force_cpu", False)),
            )
            path = study_root / "lung_lungmask.nii.gz"
            lungmask.run(image_path, path, gpu_id=gpu_id, log=progress)
            generated["masks"]["lung_lungmask"] = path
        except Exception as exc:  # noqa: BLE001 - independent QC model may be unavailable
            generated["errors"]["lungmask"] = f"{type(exc).__name__}: {exc}"
            if progress:
                progress(f"mask=lung_lungmask unavailable reason={type(exc).__name__}: {exc}")

    rows = build_rows()
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
        preview_png = _write_study_preview(
            rows,
            image_path=image_path,
            run_dir=run_dir,
            patient_id=patient_id,
            study_id=study_id,
            segmentation_config=segmentation_config,
            note="",
            pe_present=pe_present,
            progress=progress,
        )
        for item in rows:
            item["preview_png"] = preview_png
    atomic_write_json(
        state_path,
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "settings_fingerprint": None if settings_fingerprint is None else str(settings_fingerprint),
            "patient_id": patient_id,
            "study_id": study_id,
            "rows": rows,
        },
    )
    if progress:
        available = sum(bool(item.get("mask_path")) for item in rows)
        empty = sum(bool(item.get("mask_path")) and not int(item.get("voxel_count") or 0) for item in rows)
        progress(
            f"segmentation study={study_id} patient={patient_id} "
            f"status=completed masks={available}/{len(rows)} empty={empty}"
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
    # Fails loudly when the run directory was built with different mask settings.
    fingerprint = guard_resume_settings(
        run_dir, resume_settings(segmentation_config), stage="segmentation"
    )

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
            settings_fingerprint=fingerprint,
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
    partial_studies = sorted({
        str(item["study_id"]) for item in results
        if item["anatomy"] != "lung_lungmask" and item["status"] in {"FAIL", "UNAVAILABLE"}
    })
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
            "partially_failed": len(partial_studies),
            "partial_failure_study_ids": partial_studies,
        },
        "failures": failures,
        # Masks that exist but contain no voxel, and voxel-identical mask pairs (study/anatomy).
        "empty_masks": [
            f"{item['study_id']}/{item['anatomy']}"
            for item in results
            if item.get("mask_path") and not int(item.get("voxel_count") or 0)
        ],
        "duplicate_masks": sorted(
            f"{item['study_id']}/{item['anatomy']}={item['duplicate_of']}"
            for item in results
            if item.get("duplicate_of")
        ),
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
