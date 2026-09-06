from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from source.engine.experiment import atomic_write_json
from source.imaging.nifti import load_nifti, mask_qc, same_geometry, save_binary_mask
from source.imaging.preview import write_overlay_preview

from .masks import body_mask_from_hu, dilate_mask, subtract_masks, union_masks
from .random_controls import matched_random_control, stable_control_seed

SOURCE_OK = {"PASS", "SUSPICIOUS"}
ROI_OPERATIONS = {
    "ROI1": "KEEP_ONLY",
    "ROI2": "KEEP_ONLY",
    "ROI3": "REMOVE_ROI",
    "ROI4": "KEEP_ONLY",
    "ROI5": "KEEP_ONLY",
    "ROI6": "KEEP_ONLY",
    "ROI7": "REMOVE_ROI",
    "ROI8": "KEEP_ONLY",
}
STATE_SCHEMA_VERSION = 4


def _cached_rows(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    if any(row.get("roi_path") and not Path(str(row["roi_path"])).is_file() for row in rows):
        return None
    return [dict(row) for row in rows]


def _source_reason(names: Sequence[str], anatomy: Mapping[str, Mapping[str, Any]]) -> str | None:
    problems: list[str] = []
    for name in names:
        row = anatomy.get(name)
        if row is None:
            problems.append(f"{name}:missing_manifest_row")
        elif str(row.get("status")) not in SOURCE_OK:
            problems.append(f"{name}:{row.get('status')}:{row.get('reason')}")
        elif not row.get("mask_path") or not Path(str(row["mask_path"])).is_file():
            problems.append(f"{name}:missing_mask_file")
    return ";".join(problems) or None


def _recipe(
    roi_id: str,
    definition: str,
    *,
    masking_policy: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    approximation_note: str | None = None,
) -> dict[str, Any]:
    operation = ROI_OPERATIONS[roi_id]
    formula = (
        "x * M + replacement(x,M) * (1-M)"
        if operation == "KEEP_ONLY"
        else "x * (1-M) + replacement(x,M) * M"
    )
    return {
        "roi_id": roi_id,
        "operation": operation,
        "mask_definition": definition,
        "formula": formula,
        "masking_policy": dict(masking_policy),
        "parameters": dict(parameters or {}),
        "approximation_note": approximation_note,
        "interpretation": (
            "Measures predictive information retained in the selected input"
            if operation == "KEEP_ONLY"
            else "Measures paired prediction change after approximate regional erasure"
        ),
    }


def _unavailable(
    base: Mapping[str, Any],
    roi_id: str,
    reason: str,
    sources: Sequence[str],
    recipe: Mapping[str, Any],
    *,
    control_for: str | None = None,
) -> dict[str, Any]:
    return {
        **base,
        "roi_id": roi_id,
        "control_for": control_for,
        "operation": ROI_OPERATIONS[roi_id],
        "status": "UNAVAILABLE",
        "feasible": False,
        "reason": reason,
        "roi_path": None,
        "source_anatomies": json.dumps(list(sources)),
        "source_mask_paths": json.dumps([]),
        "source_statuses": json.dumps([]),
        "source_fallbacks": json.dumps([]),
        "recipe": json.dumps(dict(recipe), sort_keys=True),
        "qc_flags": json.dumps([reason]),
    }


def _process_study(
    study_rows: Sequence[Mapping[str, Any]],
    *,
    run_dir: Path,
    roi_config: Mapping[str, Any],
    seed: int,
    preview: bool,
    source_segmentation_run: str,
    source_segmentation_manifest: Path,
) -> list[dict[str, Any]]:
    first = study_rows[0]
    patient_id = str(first.get("patient_id") or "")
    study_id = str(first.get("study_id") or "")
    image_path = Path(str(first.get("image_path") or ""))
    if not patient_id or not study_id or not image_path.is_file():
        raise ValueError(
            f"invalid segmentation group: patient={patient_id!r}, study={study_id!r}, image={image_path}"
        )
    state_path = run_dir / "state" / f"{study_id}.json"
    cached = _cached_rows(state_path)
    if cached is not None:
        return cached

    anatomy = {str(row.get("anatomy")): row for row in study_rows}
    basic_paths = {
        name: str(row["mask_path"])
        for name, row in anatomy.items()
        if row.get("mask_path") and Path(str(row["mask_path"])).is_file()
    }
    masking_policy = dict(
        roi_config.get("masking_policy")
        or {"type": "local_mean", "local_radius_voxels": 3}
    )
    base = {
        "patient_id": patient_id,
        "study_id": study_id,
        "split": first.get("split"),
        "image_path": str(image_path),
        "full_ctpa_path": str(image_path),
        "basic_mask_paths": json.dumps(basic_paths, sort_keys=True),
        "source_segmentation_run": source_segmentation_run,
        "source_segmentation_manifest": str(source_segmentation_manifest),
    }
    volume, reference = load_nifti(image_path)
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    output_root = run_dir / "masks" / study_id
    rows: list[dict[str, Any]] = []
    built: dict[str, np.ndarray] = {}

    def load_sources(names: Sequence[str]) -> tuple[dict[str, np.ndarray] | None, str | None]:
        problem = _source_reason(names, anatomy)
        if problem:
            return None, problem
        values: dict[str, np.ndarray] = {}
        for name in names:
            array, image = load_nifti(str(anatomy[name]["mask_path"]))
            if not same_geometry(image, reference):
                return None, f"{name}:geometry_mismatch"
            values[name] = np.asarray(array, dtype=bool)
        return values, None

    body_values, body_problem = load_sources(("body",))
    if body_values is not None:
        body = body_values["body"]
        body_source = "TotalSegmentator body mask"
    else:
        body = body_mask_from_hu(volume, float(roi_config.get("body_threshold_hu", -900)))
        body_source = f"HU-threshold fallback ({body_problem})"
    wall_values, _ = load_sources(("body_wall",))
    body_wall = wall_values["body_wall"] if wall_values is not None else None

    def store(
        roi_id: str,
        mask: np.ndarray,
        sources: Sequence[str],
        recipe: Mapping[str, Any],
        *,
        control_for: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        suffix = f"_for_{control_for}" if control_for else ""
        path = save_binary_mask(mask, reference, output_root / f"{roi_id}{suffix}.nii.gz")
        qc = mask_qc(path, image_path)
        source_rows = [anatomy[name] for name in sources if name in anatomy]
        source_statuses = [str(item.get("status")) for item in source_rows]
        status = (
            "FAIL"
            if qc["status"] == "FAIL"
            else "SUSPICIOUS"
            if "SUSPICIOUS" in source_statuses
            else "PASS"
        )
        body_contained = bool(np.all(~np.asarray(mask, dtype=bool) | body))
        flags: list[str] = []
        if qc.get("reason") != "basic_qc_pass":
            flags.extend(str(qc.get("reason") or "").split(";"))
        if not body_contained:
            flags.append("outside_body")
            if status == "PASS":
                status = "SUSPICIOUS"
        fallbacks = [str(item.get("fallback")) for item in source_rows if item.get("fallback")]
        item = {
            **base,
            "roi_id": roi_id,
            "control_for": control_for,
            "operation": ROI_OPERATIONS[roi_id],
            "status": status,
            "feasible": status in SOURCE_OK,
            "reason": ";".join(flags) or "basic_qc_pass",
            "roi_path": str(path),
            "source_anatomies": json.dumps(list(sources)),
            "source_mask_paths": json.dumps([str(item["mask_path"]) for item in source_rows]),
            "source_statuses": json.dumps(source_statuses),
            "source_fallbacks": json.dumps(fallbacks),
            "recipe": json.dumps(dict(recipe), sort_keys=True),
            "qc_flags": json.dumps(flags),
            "voxel_count": qc.get("voxel_count"),
            "volume_ml": qc.get("volume_ml"),
            "component_count": qc.get("component_count"),
            "shape": json.dumps(qc.get("shape")),
            "spacing": json.dumps(qc.get("spacing")),
            "orientation": json.dumps(qc.get("orientation")),
            "affine": json.dumps(qc.get("affine")),
            "geometry_match": True,
            "body_contained": body_contained,
            **dict(extra or {}),
        }
        if preview:
            preview_path = run_dir / "previews" / study_id / f"{roi_id}{suffix}_overlay.png"
            write_overlay_preview(
                image_path,
                path,
                preview_path,
                study_id=study_id,
                anatomy=f"{roi_id}/{control_for}" if control_for else roi_id,
                source="derived pseudo-anatomy recipe",
                status=status,
            )
            item["preview_path"] = str(preview_path)
        rows.append(item)
        built[roi_id if control_for is None else f"{roi_id}:{control_for}"] = mask
        return item

    specifications: list[
        tuple[str, tuple[str, ...], str, Callable[[dict[str, np.ndarray]], np.ndarray], dict[str, Any], str | None]
    ] = [
        (
            "ROI1",
            ("heart", "mediastinum"),
            "heart UNION mediastinum",
            lambda values: union_masks(values["heart"], values["mediastinum"]),
            {},
            None,
        ),
        (
            "ROI2",
            ("strict_heart",),
            "strict_heart (myocardium + four chambers; excludes PA/aorta when high-res masks exist)",
            lambda values: values["strict_heart"],
            {},
            "May use the generic-heart fallback recorded by the source segmentation run "
            "when chamber masks are unavailable.",
        ),
        (
            "ROI3",
            ("central_pa",),
            "physical-space dilation of approximate central_pa",
            lambda values: dilate_mask(
                values["central_pa"], spacing, float(roi_config.get("central_pa_dilation_mm", 2.0))
            ),
            {"dilation_mm": float(roi_config.get("central_pa_dilation_mm", 2.0))},
            "The TotalSegmentator pulmonary-artery boundary is approximate; this is not perfect clot removal.",
        ),
        (
            "ROI4",
            ("pa_tree",),
            "central_pa UNION intrapulmonary lung_arteries",
            lambda values: values["pa_tree"],
            {},
            "Union of two model outputs; continuity and distal-vessel coverage are not guaranteed.",
        ),
        (
            "ROI5",
            ("lung",),
            "whole lung mask retaining native intrapulmonary vessels",
            lambda values: values["lung"],
            {},
            None,
        ),
    ]
    airway_subtraction = bool(roi_config.get("subtract_airways", False))
    roi6_sources = (
        "lung",
        "lung_arteries",
        "lung_veins",
        *(("airways",) if airway_subtraction else ()),
    )
    specifications.append(
        (
            "ROI6",
            roi6_sources,
            "lung MINUS (lung_arteries UNION lung_veins"
            + (" UNION large airways)" if airway_subtraction else ")"),
            lambda values: subtract_masks(
                values["lung"],
                values["lung_arteries"],
                values["lung_veins"],
                *((values["airways"],) if airway_subtraction else ()),
            ),
            {"subtract_large_airways": airway_subtraction},
            (
                "Vessel-and-airway-reduced parenchyma; microscopic vessels are not claimed removed."
                if airway_subtraction
                else "Vessel-reduced parenchyma; microscopic vessels are not claimed removed."
            ),
        )
    )

    for roi_id, sources, definition, operation, parameters, note in specifications:
        recipe = _recipe(
            roi_id,
            definition,
            masking_policy=masking_policy,
            parameters=parameters,
            approximation_note=note,
        )
        values, problem = load_sources(sources)
        if values is None:
            rows.append(_unavailable(base, roi_id, str(problem), sources, recipe))
            continue
        store(roi_id, operation(values), sources, recipe)

    configured_hilar = tuple(str(value) for value in (roi_config.get("hilar_mask_names") or ()))
    hilar_names = configured_hilar or ("hilar_vessels",)
    roi7_sources = ("heart", *hilar_names)
    roi7_recipe = _recipe(
        "ROI7",
        "heart UNION " + " UNION ".join(hilar_names),
        masking_policy=masking_policy,
        approximation_note=(
            "Hilar-vessel masks are proximity-derived approximations unless an externally validated mask is configured."
        ),
    )
    roi7_values, roi7_problem = load_sources(roi7_sources)
    if roi7_values is None:
        rows.append(_unavailable(base, "ROI7", str(roi7_problem), roi7_sources, roi7_recipe))
    else:
        store(
            "ROI7",
            union_masks(*(roi7_values[name] for name in roi7_sources)),
            roi7_sources,
            roi7_recipe,
        )

    for control_for, label in (("ROI2", "heart"), ("ROI4", "pa"), ("ROI6", "lung")):
        target = built.get(control_for)
        random_recipe = _recipe(
            "ROI8",
            f"deterministic {label}-volume-matched control inside body/body-wall and outside {control_for}",
            masking_policy=masking_policy,
            parameters={
                "control_for": control_for,
                "exclusion_margin_mm": float(roi_config.get("random_exclusion_margin_mm", 2.0)),
            },
            approximation_note="Physical volume is matched exactly; control shape is not claimed matched.",
        )
        if target is None:
            rows.append(
                _unavailable(
                    base,
                    "ROI8",
                    f"source {control_for} unavailable",
                    (control_for,),
                    random_recipe,
                    control_for=control_for,
                )
            )
            continue
        control_seed = stable_control_seed(seed, patient_id, study_id, control_for)
        try:
            control, metadata = matched_random_control(
                target=target,
                body=body,
                body_wall=body_wall,
                spacing=spacing,
                seed=control_seed,
                exclusion_margin_mm=float(roi_config.get("random_exclusion_margin_mm", 2.0)),
            )
        except ValueError as exc:
            rows.append(
                _unavailable(
                    base,
                    "ROI8",
                    str(exc),
                    (control_for,),
                    random_recipe,
                    control_for=control_for,
                )
            )
            continue
        source_item = next(row for row in rows if row["roi_id"] == control_for)
        anatomy[control_for] = {
            **base,
            "mask_path": source_item["roi_path"],
            "status": source_item["status"],
            "fallback": None,
        }
        store(
            "ROI8",
            control,
            (control_for,),
            random_recipe,
            control_for=control_for,
            extra={"seed": control_seed, "body_source": body_source, **metadata},
        )

    atomic_write_json(
        state_path,
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "patient_id": patient_id,
            "study_id": study_id,
            "rows": rows,
        },
    )
    return rows


def compact_roi_manifest(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert normalized ROI QC rows to one canonical recipe record per study."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("patient_id") or ""), str(row.get("study_id") or ""))].append(row)
    output: list[dict[str, Any]] = []
    for (patient_id, study_id), items in sorted(grouped.items()):
        first = items[0]
        compact: dict[str, Any] = {
            "patient_id": patient_id,
            "study_id": study_id,
            "split": first.get("split"),
            "full_ctpa_path": first.get("full_ctpa_path") or first.get("image_path"),
            "basic_mask_paths": first.get("basic_mask_paths"),
            "source_segmentation_run": first.get("source_segmentation_run"),
            "source_segmentation_manifest": first.get("source_segmentation_manifest"),
        }
        qc_flags: dict[str, Any] = {}
        feasibility: dict[str, bool] = {}
        for item in items:
            roi_id = str(item["roi_id"])
            control_for = str(item.get("control_for") or "")
            suffix = (
                "_heart"
                if control_for == "ROI2"
                else "_pa"
                if control_for == "ROI4"
                else "_lung"
                if control_for == "ROI6"
                else ""
            )
            key = f"{roi_id}{suffix}"
            compact[f"{key}_mask_path"] = item.get("roi_path")
            compact[f"{key}_recipe"] = item.get("recipe")
            if item.get("seed") is not None:
                compact[f"{key}_seed"] = item.get("seed")
            try:
                flags = json.loads(str(item.get("qc_flags") or "[]"))
            except json.JSONDecodeError:
                flags = [str(item.get("qc_flags"))]
            qc_flags[key] = {
                "status": item.get("status"),
                "reason": item.get("reason"),
                "flags": flags,
            }
            feasibility[key] = bool(item.get("feasible", False))
        compact["qc_flags"] = json.dumps(qc_flags, sort_keys=True)
        compact["feasibility_flags"] = json.dumps(feasibility, sort_keys=True)
        output.append(compact)
    return output


def build_roi_dataset(
    segmentation_rows: Sequence[Mapping[str, Any]],
    *,
    run_dir: Path,
    roi_config: Mapping[str, Any],
    maximum_cases: int | None,
    seed: int,
    source_segmentation_run: str,
    source_segmentation_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in segmentation_rows:
        grouped[str(row.get("study_id") or "")].append(row)
    selected = sorted(grouped.items())
    if maximum_cases is not None:
        selected = selected[:maximum_cases]
    workers = max(1, int(roi_config.get("workers", 1)))
    preview_cases = max(0, int(roi_config.get("preview_cases", 5)))
    output: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_study,
                rows,
                run_dir=run_dir,
                roi_config=roi_config,
                seed=seed,
                preview=index < preview_cases,
                source_segmentation_run=source_segmentation_run,
                source_segmentation_manifest=source_segmentation_manifest,
            ): study_id
            for index, (study_id, rows) in enumerate(selected)
        }
        for future in as_completed(futures):
            try:
                output.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve per-study failure accounting
                failures.append({"study_id": futures[future], "reason": f"{type(exc).__name__}: {exc}"})
    output.sort(
        key=lambda row: (str(row["study_id"]), str(row["roi_id"]), str(row.get("control_for") or ""))
    )
    counts = Counter((str(row["roi_id"]), str(row["status"])) for row in output)
    evaluation = {
        "studies": {
            "requested": len(selected),
            "processed": len(selected) - len(failures),
            "failed": len(failures),
        },
        "failures": failures,
        "source_segmentation_run": source_segmentation_run,
        "source_segmentation_manifest": str(source_segmentation_manifest),
        "scientific_interpretation": (
            "ROI performance quantifies retained predictive information; counterfactual deltas quantify "
            "prediction change after approximate erasure. Neither establishes a causal biological mechanism."
        ),
        "roi": {
            roi_id: {
                status: counts[(roi_id, status)]
                for status in ("PASS", "SUSPICIOUS", "FAIL", "UNAVAILABLE")
            }
            for roi_id in ROI_OPERATIONS
        },
        "definitions": {
            "ROI1": "heart UNION mediastinum; KEEP_ONLY",
            "ROI2": "strict heart; KEEP_ONLY; explicit generic-heart fallback provenance",
            "ROI3": "dilated approximate central_pa; REMOVE_ROI",
            "ROI4": "central_pa UNION lung_arteries; KEEP_ONLY",
            "ROI5": "whole lung retaining vessels; KEEP_ONLY",
            "ROI6": "lung MINUS vessels, optionally large airways; KEEP_ONLY",
            "ROI7": "heart UNION approximate/configured hilar vessels; REMOVE_ROI",
            "ROI8": "deterministic non-overlapping body/body-wall controls volume-matched to ROI2/4/6",
        },
    }
    return output, evaluation
