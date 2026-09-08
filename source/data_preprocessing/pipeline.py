"""One dataset build, driven entirely by a dataset profile.

`test_500_sample` and `full_inspect` execute this function with the same eligibility,
integrity, adjudication, manifest and preprocessing blocks; the only difference between
them is the `sampling` block. That is the whole point: a pilot run rehearses the code
path the full run will take, so nothing is "pilot-only" or "full-only".

    build_dataset(profile, paths)
        -> <derived>/datasets/<profile>/
             manifests/{ctpa,diagnosis,prognosis,paired_reports,reports}.csv
             data_quality.{md,json}  compact, de-identified readiness report
             audit/                 detailed exclusion, integrity and cache-QC findings
             dataset.json          full provenance: profile, rules, sampling plan, hashes
"""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from source.data.paths import ProjectPaths

from .adjudication import patient_labels
from .ehr import EhrBuildError, build_ehr_readiness
from .filters import apply_eligibility
from .integrity import check_volumes, integrity_summary
from .leakage import audit_split_integrity, load_excluded_patients, require_no_leakage
from .manifests import build_manifests
from .pesi import PesiBuildError, build_pesi_artifacts
from .quality import build_data_quality, cohort_step, render_data_quality_markdown
from .sampling import sample_patients
from .sources import InspectSource, StudyRecord
from .volumes import PreprocessingSpec, preprocess_study, validate_cache_entry


class DatasetBuildError(RuntimeError):
    pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    temporary.replace(path)
    return path


def resolve_release_root(paths: ProjectPaths, profile: Mapping[str, Any]) -> Path:
    source = dict(profile.get("source") or {})
    explicit = source.get("root")
    if explicit:
        return Path(str(explicit))
    if paths.raw_inspect_root is None:
        raise DatasetBuildError(
            "raw INSPECT root is not configured; set PE_CLOUD_ROOT or paths.raw_inspect"
        )
    modality = str(source.get("modality") or "CT")
    release = str(source.get("release") or "full")
    return paths.raw_inspect_root / modality / release


def dataset_output_root(paths: ProjectPaths, name: str) -> Path:
    if paths.derived_root is None:
        raise DatasetBuildError("derived data root is not configured; set PE_CLOUD_ROOT")
    return paths.derived_root / "datasets" / name


def build_dataset(
    profile: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    output_root: str | Path | None = None,
    max_cases: int | None = None,
    allow_full: bool = False,
    preprocess: bool = False,
    overwrite: bool = False,
    check_files: bool | None = None,
) -> dict[str, Any]:
    """Build one dataset profile end to end. Returns the provenance payload it writes.

    ``max_cases`` limits the cohort to the first N eligible studies (a smoke test);
    ``allow_full`` is required to build the whole profile, mirroring the explicit-scope
    rule the rest of the pipeline uses so a full build is never accidental.
    """
    name = str((profile.get("profile") or {}).get("name") or "").strip()
    if not name:
        raise DatasetBuildError("dataset profile has no profile.name")
    if max_cases is None and not allow_full:
        raise DatasetBuildError(
            f"building dataset {name!r} needs an explicit scope: pass max_cases=N or allow_full=True"
        )

    release_root = resolve_release_root(paths, profile)
    source = InspectSource(
        release_root,
        volume_suffix=str((profile.get("source") or {}).get("volume_suffix") or ".nii.gz"),
    )
    records = source.records()
    official_splits = {record.study_id: record.split for record in records}
    # One row per step that can drop a case, so data_quality.md can report exactly how many
    # samples survive each stage instead of leaving the reader to reconcile three reports.
    funnel = [cohort_step("release", "studies joined from the read-only INSPECT release", records)]

    governance = list((profile.get("governance") or {}).get("exclude_patients_from") or [])
    excluded_patients = load_excluded_patients(governance) if governance else set()
    if excluded_patients:
        funnel.append(
            cohort_step(
                "governance",
                "patients on an exclusion registry, removed whole",
                [record for record in records if record.patient_id not in excluded_patients],
            )
        )

    eligibility = dict(profile.get("eligibility") or {})
    resolved_check_files = (
        bool(eligibility.get("check_files", True)) if check_files is None else bool(check_files)
    )
    eligible, ledger, eligibility_report = apply_eligibility(
        records,
        eligibility.get("rules"),
        excluded_patients=excluded_patients,
        check_files=resolved_check_files,
    )
    if not eligible:
        raise DatasetBuildError(f"dataset {name!r} has no eligible studies after filtering")
    funnel.append(
        cohort_step("eligibility", "missing inputs and out-of-range acquisitions", eligible)
    )

    integrity = dict(profile.get("integrity") or {})
    integrity_limit = integrity.get("limit")
    if max_cases is not None:
        integrity_limit = int(max_cases) if integrity_limit is None else min(int(integrity_limit), int(max_cases))
    usable, checks = check_volumes(
        eligible,
        level=str(integrity.get("level") or "path"),
        limit=None if integrity_limit is None else int(integrity_limit),
        minimum_slices=int(integrity.get("minimum_slices", 2)),
    )
    for check in checks:
        if check.status != "PASS":
            record = next(item for item in eligible if item.study_id == check.study_id)
            ledger.record(record, f"ct_{check.status.lower()}", check.detail)

    funnel.append(cohort_step("ct_integrity", "missing or unreadable CT volumes", usable))

    adjudication_columns = tuple(
        (profile.get("adjudication") or {}).get("columns")
        or ("pe_positive_nlp", "pe_acute", "pe_subsegmentalonly", "1_month_mortality")
    )
    labels = patient_labels(usable, adjudication_columns)

    sampling = dict(profile.get("sampling") or {})
    sampling_plan = None
    cohort: list[StudyRecord] = list(usable)
    if bool(sampling.get("enabled")):
        selected, sampling_plan = sample_patients(
            labels,
            seed=int(sampling.get("seed", 42)),
            total_patients=sampling.get("total_patients"),
            per_split=sampling.get("per_split"),
            strata=tuple(sampling.get("strata") or adjudication_columns),
            balance_column=sampling.get("balance_column"),
            positive_fraction=sampling.get("positive_fraction"),
        )
        # Patient-level: a selected patient keeps every one of their studies.
        cohort = [record for record in usable if record.patient_id in selected]
        funnel.append(
            cohort_step("sampling", f"profile sampling ({name})", cohort)
        )
    if max_cases is not None:
        keep: list[StudyRecord] = []
        patients: set[str] = set()
        for record in cohort:
            if len(patients) >= int(max_cases) and record.patient_id not in patients:
                continue
            patients.add(record.patient_id)
            keep.append(record)
        cohort = keep
        funnel.append(cohort_step("scope", f"explicit run scope max_cases={max_cases}", cohort))

    # Check the selected cohort before doing expensive cache writes. A second audit below
    # protects the final cohort after per-study preprocessing/cache QC has removed failures.
    selected_audit = audit_split_integrity(
        cohort, official_splits, require_all_splits=bool(profile.get("require_all_splits", max_cases is None))
    )
    require_no_leakage(selected_audit)

    destination = Path(output_root) if output_root else dataset_output_root(paths, name)
    if destination.exists() and not overwrite and any(destination.iterdir()):
        existing = destination / "dataset.json"
        if existing.is_file():
            raise DatasetBuildError(
                f"dataset {name!r} already exists at {destination}; pass overwrite=True to rebuild it"
            )
    destination.mkdir(parents=True, exist_ok=True)

    spec = PreprocessingSpec.from_mapping(profile.get("preprocessing"))
    image_paths: dict[str, str] = {}
    preprocessing_report: dict[str, Any] = {
        "enabled": bool(preprocess),
        "fingerprint": spec.fingerprint(),
        "spec": spec.as_dict(),
    }
    if preprocess:
        cache_dir = destination / "volumes"
        written, failures, cache_qc_failures = 0, [], []
        cache_qc = dict(profile.get("cache_qc") or {})
        cache_qc_enabled = bool(cache_qc.get("enabled", True))
        cache_qc_limit = cache_qc.get("limit")
        if cache_qc_limit is not None and int(cache_qc_limit) < 1:
            raise DatasetBuildError("cache_qc.limit must be positive or null")
        cache_qc_checked = 0
        cache_qc_candidates = 0
        for record in cohort:
            try:
                row = preprocess_study(record.study_id, record.image_path, cache_dir, spec, overwrite=overwrite)
            except Exception as exc:  # noqa: BLE001 - a batch build records and continues
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"study_id": record.study_id, "error": error})
                ledger.record(record, "ct_preprocessing_failed", error)
                continue
            cache_qc_candidates += 1
            if cache_qc_enabled and (cache_qc_limit is None or cache_qc_checked < int(cache_qc_limit)):
                cache_qc_checked += 1
                errors = validate_cache_entry(row["preprocessed_path"], spec)
                if errors:
                    detail = "; ".join(errors)
                    cache_qc_failures.append({"study_id": record.study_id, "errors": errors})
                    ledger.record(record, "ct_cache_qc_failed", detail)
                    continue
            image_paths[record.study_id] = row["preprocessed_path"]
            written += row["status"] == "written"
        preprocessing_report.update(
            {"cache_dir": str(cache_dir), "written": written, "cached": len(image_paths) - written,
             "failures": failures[:200], "failure_count": len(failures),
             "cache_qc": {
                 "enabled": cache_qc_enabled,
                 "checked": cache_qc_checked,
                 "passed": cache_qc_checked - len(cache_qc_failures),
                 "failed": len(cache_qc_failures),
                 "skipped": cache_qc_candidates - cache_qc_checked,
                 "failures": cache_qc_failures[:200],
             }}
        )
        cohort = [record for record in cohort if record.study_id in image_paths]
        if not cohort:
            raise DatasetBuildError("every study failed preprocessing; nothing to write")
        funnel.append(
            cohort_step("preprocessing", "CT preprocessing and cache QC failures", cohort)
        )

    audit = audit_split_integrity(
        cohort, official_splits, require_all_splits=bool(profile.get("require_all_splits", max_cases is None))
    )
    require_no_leakage(audit)

    clinical_dir = destination / "clinical"
    try:
        ehr = build_ehr_readiness(
            cohort,
            release_root=release_root,
            cache_root=paths.cache_root,
            output_dir=clinical_dir,
            config=profile.get("ehr"),
            code_root=paths.code_root,
        )
    except EhrBuildError as exc:
        raise DatasetBuildError(f"EHR readiness failed: {exc}") from exc

    try:
        pesi = build_pesi_artifacts(
            cohort,
            output_dir=clinical_dir,
            config=profile.get("pesi"),
            ehr_metadata=ehr.metadata,
            code_root=paths.code_root,
        )
    except PesiBuildError as exc:
        raise DatasetBuildError(f"PESI readiness failed: {exc}") from exc

    clinical_features: dict[str, dict[str, Any]] = {}
    clinical_columns = (*ehr.feature_columns, *pesi.feature_columns)
    if len(set(clinical_columns)) != len(clinical_columns):
        raise DatasetBuildError("EHR and PESI feature columns overlap")
    for record in cohort:
        clinical_features[record.study_id] = {
            **dict(ehr.by_study.get(record.study_id) or {}),
            **dict(pesi.by_study.get(record.study_id) or {}),
        }

    manifest_config = dict(profile.get("manifests") or {})
    manifest_summary = build_manifests(
        cohort,
        destination / "manifests",
        label_map=manifest_config.get("label_map"),
        prognosis_label=str(manifest_config.get("prognosis_label") or "1_month_mortality"),
        prognosis_conditions=manifest_config.get("prognosis_conditions"),
        image_paths=image_paths or None,
        include_report_text=bool(manifest_config.get("include_report_text", True)),
        clinical_features=clinical_features,
        clinical_columns=clinical_columns,
    )

    audit_dir = destination / "audit"
    _write_rows(
        audit_dir / "exclusions.csv",
        ledger.entries,
        ["patient_id", "study_id", "impression_id", "split", "rule", "detail"],
    )
    integrity_payload = integrity_summary(checks)
    _write_json(audit_dir / "integrity.json", integrity_payload)
    _write_json(audit_dir / "split_audit.json", audit.as_dict())
    _write_json(audit_dir / "cache_qc.json", dict(preprocessing_report.get("cache_qc") or {"enabled": False}))

    eligibility_payload = eligibility_report.as_dict()
    eligibility_payload["excluded_studies"] = len(ledger.entries)
    eligibility_payload["excluded_by_rule"] = ledger.counts()
    sampling_payload = sampling_plan.as_dict() if sampling_plan else {"enabled": False}
    if sampling_plan:
        sampling_payload["enabled"] = True

    payload: dict[str, Any] = {
        "profile": dict(profile.get("profile") or {}),
        "source": source.provenance(),
        "scope": {"max_cases": max_cases, "allow_full": bool(allow_full)},
        "eligibility": eligibility_payload,
        "governance": {"exclusion_sources": [str(item) for item in governance],
                       "excluded_patients": len(excluded_patients)},
        "integrity": {"level": str(integrity.get("level") or "path"), "limit": integrity_limit,
                      **integrity_payload},
        "adjudication": {"columns": list(adjudication_columns), "patients": len(labels)},
        "sampling": sampling_payload,
        "split_audit": audit.as_dict(),
        "preprocessing": preprocessing_report,
        "clinical": {
            "ehr": dict(ehr.metadata),
            "pesi": dict(pesi.metadata),
        },
        "manifests": manifest_summary,
        "cohort": {
            "studies": len(cohort),
            "patients": len({record.patient_id for record in cohort}),
        },
        "output_root": str(destination),
    }
    quality = build_data_quality(
        profile=payload["profile"],
        release_root=release_root,
        records=cohort,
        adjudication_columns=adjudication_columns,
        eligibility=eligibility_payload,
        integrity=payload["integrity"],
        preprocessing=preprocessing_report,
        split_audit=audit.as_dict(),
        sampling=sampling_payload,
        ehr=dict(ehr.metadata),
        pesi=pesi.metadata,
        cohort_funnel=[*funnel, cohort_step("final", "the cohort written to the manifests", cohort)],
    )
    _write_json(destination / "data_quality.json", quality)
    quality_markdown = destination / "data_quality.md"
    _write_text(quality_markdown, render_data_quality_markdown(quality))
    payload["quality"] = {
        "json": str(destination / "data_quality.json"),
        "markdown": str(quality_markdown),
        "audit_dir": str(audit_dir),
    }
    _write_json(destination / "dataset.json", payload)
    return payload
