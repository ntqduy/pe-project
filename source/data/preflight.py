from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from source.data.manifests import (
    ManifestError,
    audit_manifest,
    audit_report_table,
    audit_silver_table,
    patient_ids_for_splits,
    read_rows,
)
from source.data.paths import PathConfigurationError, ProjectPaths
from source.engine.checkpoint import CheckpointError, inspect_checkpoint
from source.engine.experiment import OutputManager, verify_writable_directory


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[Check, ...]
    manifest: dict[str, Any] | None
    source_checkpoint: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return all(check.status in {"PASS", "SKIP"} for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
            "manifest": self.manifest,
            "source_checkpoint": self.source_checkpoint,
        }


def _exists(checks: list[Check], group: str, name: str, path: Path | None, required: bool) -> None:
    if path is not None and path.exists():
        checks.append(Check(group, name, "PASS", str(path)))
    elif required:
        checks.append(Check(group, name, "FAIL", str(path) if path else "not configured"))
    else:
        checks.append(Check(group, name, "SKIP", str(path) if path else "not configured"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_placeholder(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "none",
        "null",
        "unspecified",
        "todo",
        "tbd",
        "replace_me",
        "placeholder",
    }


def _checksum_check(
    checks: list[Check],
    *,
    group: str,
    name: str,
    path: Path | None,
    expected: Any,
    allow_auto: bool = False,
) -> str | None:
    if path is None or not path.is_file():
        checks.append(Check(group, name, "FAIL", "artifact unavailable for checksum validation"))
        return None
    actual = _sha256(path)
    configured = str(expected or "").strip()
    if allow_auto and configured.lower() == "auto":
        checks.append(Check(group, name, "PASS", actual))
    elif len(configured) != 64:
        checks.append(Check(group, name, "FAIL", "explicit SHA-256 is not configured"))
    elif configured.lower() != actual.lower():
        checks.append(Check(group, name, "FAIL", f"expected={configured} actual={actual}"))
    else:
        checks.append(Check(group, name, "PASS", actual))
    return actual


def _artifact_path(
    paths: ProjectPaths, value: Any, *, output: bool = False, profile: str | None = None
) -> Path | None:
    if value is None or not str(value).strip():
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()
    if output:
        return paths.output_asset(candidate)
    try:
        return (paths.dataset_root("full", profile) / candidate).resolve()
    except PathConfigurationError:
        return paths.code_asset(candidate)


def _is_inspect_dataset(name: str) -> bool:
    """Every active dataset profile is an INSPECT cohort, so the leakage guard must
    recognise the profile names as well as the historical bare ``inspect``."""
    normalized = str(name or "").strip().lower()
    if normalized in {"inspect", "stanford_inspect"}:
        return True
    try:
        from source.dataset import ACTIVE_PROFILES
    except Exception:  # noqa: BLE001 - preflight must not fail on an import
        return "inspect" in normalized
    return normalized in {value.lower() for value in ACTIVE_PROFILES} or "inspect" in normalized


def checkpoint_lineage_errors(
    config: Mapping[str, Any],
    source_lineage: Mapping[str, Any],
    manifest_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return configuration/checkpoint incompatibilities, including obvious fold leakage."""
    errors: list[str] = []
    configured = dict(config.get("lineage") or {})
    data = dict(config.get("data") or {})
    task = dict(config.get("task") or {})
    experiment = dict(config.get("experiment") or {})
    expected_experiment = configured.get("source_experiment")
    if expected_experiment and source_lineage.get("experiment_id") != expected_experiment:
        errors.append(
            "source experiment mismatch: "
            f"configured={expected_experiment} checkpoint={source_lineage.get('experiment_id')}"
        )
    expected_backbone = configured.get("backbone") or (config.get("model") or {}).get("backbone")
    if expected_backbone and str(source_lineage.get("backbone") or "").lower() != str(expected_backbone).lower():
        errors.append(
            "source backbone mismatch: "
            f"configured={expected_backbone} checkpoint={source_lineage.get('backbone')}"
        )

    effective_stage = str(experiment.get("stage") or "")
    if effective_stage in {"ablation", "counterfactual", "roi_student"}:
        effective_stage = str(task.get("base_stage") or "")
    source_dataset = str(source_lineage.get("dataset") or "").lower()
    source_stage = str(source_lineage.get("stage") or "").lower()
    source_task = str(source_lineage.get("task") or "").lower()
    source_supervision = str(source_lineage.get("supervision_type") or "").lower()
    inspect_supervised_diagnosis = (
        _is_inspect_dataset(source_dataset)
        and (source_stage == "diagnosis" or "diagnos" in source_task)
        and source_supervision not in {"none", "public", "self_supervised", "image_report_alignment"}
    )
    if effective_stage == "prognosis" and inspect_supervised_diagnosis:
        patient_column = str(data.get("patient_id_column") or "patient_id")
        split_column = str(data.get("split_column") or "split")
        evaluation_patients = patient_ids_for_splits(
            manifest_rows or [],
            ("validation", "test", "external"),
            patient_column=patient_column,
            split_column=split_column,
        )
        source_train = {
            str(value).strip()
            for value in source_lineage.get("train_patient_ids") or ()
            if str(value).strip()
        }
        if source_train:
            overlap = sorted(evaluation_patients & source_train)
            if overlap:
                errors.append(
                    "INSPECT diagnosis checkpoint trained on prognosis evaluation patients: "
                    + ", ".join(overlap[:10])
                )
        else:
            fold = data.get("fold")
            fold_safe = (
                fold is not None
                and bool(source_lineage.get("fold_aware"))
                and str(source_lineage.get("held_out_fold")) == str(fold)
            )
            if not fold_safe:
                errors.append(
                    "INSPECT-supervised diagnosis initialization for prognosis requires either "
                    "disjoint train_patient_ids provenance or fold_aware=true with a matching held_out_fold"
                )
    return errors


def run_preflight(config: Mapping[str, Any], paths: ProjectPaths) -> PreflightReport:
    checks: list[Check] = []
    manifest_rows: list[dict[str, Any]] | None = None
    source_checkpoint_payload: dict[str, Any] | None = None
    mode = str((config.get("data") or {}).get("mode"))
    checks.append(
        Check("PROJECT", "code_root", "PASS" if paths.code_root.is_dir() else "FAIL", str(paths.code_root))
    )
    try:
        output_root = paths.assert_persistent_output()
        verify_writable_directory(output_root)
        free = shutil.disk_usage(output_root).free
        checks.append(Check("OUTPUT", "persistent_write", "PASS", f"{output_root} ({free} bytes free)"))
    except (PathConfigurationError, OSError, RuntimeError) as exc:
        checks.append(Check("OUTPUT", "persistent_write", "FAIL", str(exc)))
    _exists(checks, "PROJECT", "cloud_project_root", paths.cloud_project_root, True)
    _exists(checks, "DATA", "raw_inspect_read_only", paths.raw_inspect_root, mode == "full")
    # For the dataset stage the derived tree is the *output*; the destination check below
    # covers it. Every other stage reads from it, so it has to exist already.
    building_dataset = str((config.get("experiment") or {}).get("stage") or "") == "dataset"
    _exists(checks, "DATA", "derived", paths.derived_root, mode == "full" and not building_dataset)
    profile = str((config.get("data") or {}).get("profile") or "")
    if profile and str((config.get("experiment") or {}).get("stage") or "") != "dataset":
        try:
            built = paths.dataset_root_for(config)
        except PathConfigurationError as exc:
            checks.append(Check("DATA", "dataset_profile", "FAIL", str(exc)))
        else:
            provenance = built / "dataset.json"
            _exists(checks, "DATA", "dataset_profile", provenance, True)
            if provenance.is_file():
                try:
                    payload = json.loads(provenance.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    checks.append(Check("DATA", "dataset_provenance", "FAIL", str(exc)))
                else:
                    cohort = dict(payload.get("cohort") or {})
                    fingerprint = str((payload.get("preprocessing") or {}).get("fingerprint") or "")
                    checks.append(
                        Check(
                            "DATA", "dataset_provenance", "PASS",
                            f"{profile}: {cohort.get('patients')} patients, "
                            f"{cohort.get('studies')} studies, "
                            f"preprocessing={fingerprint[:16] or 'unrecorded'}",
                        )
                    )
    stage = str((config.get("experiment") or {}).get("stage") or "")
    data_config = dict(config.get("data") or {})
    task = dict(config.get("task") or {})
    effective_stage = (
        str(task.get("base_stage") or "")
        if stage in {"ablation", "counterfactual", "roi_student"}
        else stage
    )
    report_only_contract = (
        effective_stage == "diagnosis"
        and str(task.get("architecture", "soft_moe")) == "report_only"
    )
    manifest_path = data_config.get("manifest")
    manifest_payload: dict[str, Any] | None = None
    if stage == "silver":
        report_value = (config.get("silver") or {}).get("reports")
        report_path = Path(str(report_value)) if report_value else None
        if report_path is not None and not report_path.is_absolute():
            try:
                report_path = paths.dataset_root_for(config) / report_path
            except PathConfigurationError:
                report_path = None
        _exists(checks, "DATA", "reports", report_path, True)
        if report_path is not None and report_path.is_file():
            try:
                report_audit = audit_report_table(report_path)
                checks.append(
                    Check(
                        "DATA", "report_table_schema",
                        "PASS" if report_audit.ok else "FAIL",
                        "; ".join(report_audit.errors) or f"rows={report_audit.rows}",
                    )
                )
            except ManifestError as exc:
                checks.append(Check("DATA", "report_table_schema", "FAIL", str(exc)))
    elif stage == "dataset":
        # This stage *produces* the manifests, so there is nothing to audit yet. What must
        # hold beforehand is that the profile exists, that it shares the preprocessing
        # contract with the other active profile, and that the read-only release is there.
        profile_name = str(data_config.get("profile") or "")
        try:
            from source.dataset import assert_shared_preprocessing, require_active_profile

            profile = require_active_profile(profile_name)
            checks.append(
                Check("DATA", "dataset_profile", "PASS",
                      f"{profile_name}: {str((profile.get('profile') or {}).get('scope') or '')}")
            )
            checks.append(
                Check("DATA", "shared_preprocessing", "PASS",
                      f"sha256={assert_shared_preprocessing()[:16]} across the active profiles")
            )
            release = None
            if paths.raw_inspect_root is not None:
                source_block = dict(profile.get("source") or {})
                release = paths.raw_inspect_root / str(source_block.get("modality") or "CT") / str(
                    source_block.get("release") or "full"
                )
            _exists(checks, "DATA", "inspect_release", release, True)
        except Exception as exc:  # noqa: BLE001 - report the contract failure, never crash
            checks.append(Check("DATA", "dataset_profile", "FAIL", f"{type(exc).__name__}: {exc}"))
    elif not manifest_path:
        checks.append(
            Check(
                "DATA",
                "split_manifest",
                "FAIL",
                "data.manifest is required; no split is created implicitly",
            )
        )
    else:
        manifest = Path(str(manifest_path))
        if not manifest.is_absolute():
            manifest = paths.dataset_root_for(config) / manifest
        try:
            audit = audit_manifest(
                manifest,
                label_columns=tuple(data_config.get("label_columns") or ()),
                file_column=(None if report_only_contract else data_config.get("file_column", "image_path")),
                data_root=paths.dataset_root_for(config),
                split_aliases=data_config.get("split_aliases"),
                required_columns=tuple(
                    dict.fromkeys(
                        (
                            *(() if report_only_contract else (data_config.get("file_column", "image_path"),)),
                            *tuple(data_config.get("label_columns") or ()),
                            *tuple(data_config.get("ehr_columns") or ()),
                            *tuple(data_config.get("pesi_columns") or ()),
                            *tuple((data_config.get("mask_columns") or {}).values()),
                            *tuple(
                                (config.get("alignment") or {}).get("report_embedding_columns") or ()
                                if report_only_contract
                                else ()
                            ),
                        )
                    )
                ),
                primary_target=(
                    str(task.get("primary_target"))
                    if effective_stage in {"diagnosis", "prognosis"}
                    and task.get("primary_target")
                    else None
                ),
                fold_column=data_config.get("fold_column"),
                required_splits=(
                    ("train", "validation", "test")
                    if effective_stage in {"diagnosis", "prognosis"}
                    and stage != "counterfactual"
                    else ("validation", "test") if stage == "counterfactual" else ()
                ),
            )
            manifest_payload = audit.as_dict()
            checks.append(
                Check(
                    "DATA",
                    "split_manifest",
                    "PASS" if audit.ok else "FAIL",
                    "; ".join(audit.errors) or str(manifest),
                )
            )
            if audit.ok:
                try:
                    manifest_rows = read_rows(manifest)
                except ManifestError:
                    manifest_rows = None
        except (ManifestError, PathConfigurationError) as exc:
            checks.append(Check("DATA", "split_manifest", "FAIL", str(exc)))
    model = dict(config.get("model") or {})
    backbone = str(model.get("backbone") or "")
    report_only_diagnosis = report_only_contract
    if report_only_diagnosis:
        count = len((config.get("alignment") or {}).get("report_embedding_columns") or ())
        checks.append(
            Check(
                "DATA",
                "report_embedding_contract",
                "PASS" if count > 0 else "FAIL",
                f"columns={count}",
            )
        )
    elif stage in {
        "foundation", "dapt", "alignment", "silver_encoder_adaptation",
        "diagnosis", "prognosis", "contour", "ablation", "counterfactual", "roi_student",
    }:
        uses_image_model = not (
            effective_stage == "prognosis"
            and "image" not in set(task.get("modalities") or ())
        )
        if not uses_image_model:
            checks.append(Check("MODEL", "image_backbone", "SKIP", "image modality disabled"))
        elif backbone:
            _exists(checks, "MODEL", f"{backbone}_repo", paths.code_asset(model.get("repo")), True)
            _exists(
                checks,
                "MODEL",
                f"{backbone}_checkpoint",
                paths.code_asset(model.get("checkpoint")),
                bool(model.get("load_pretrained", True)),
            )
            contract_ok = bool(
                model.get("factory")
                and model.get("output_adapter")
                and int(model.get("feature_dim") or 0) > 0
            )
            checks.append(
                Check(
                    "MODEL",
                    f"{backbone}_adapter_contract",
                    "PASS" if contract_ok else "FAIL",
                    "factory/output_adapter/feature_dim",
                )
            )
        else:
            checks.append(Check("MODEL", "backbone", "FAIL", "missing"))
    elif stage == "segmentation_validation":
        specific = dict(config.get("segmentation_finetuning") or {})
        available = bool(specific.get("available", False))
        checks.append(
            Check(
                "SUPERVISION",
                "ctpa_expert_annotations",
                "PASS" if available else "FAIL",
                (
                    "configured"
                    if available
                    else "UNAVAILABLE: expert-reviewed CTPA segmentation annotations are not configured"
                ),
            )
        )
        annotation = specific.get("annotation_manifest")
        initialization = specific.get("public_initialization_checkpoint")
        _exists(
            checks, "SUPERVISION", "annotation_manifest",
            _artifact_path(paths, annotation, profile=(config.get("data") or {}).get("profile")),
            available,
        )
        _exists(
            checks, "MODEL", "public_initialization_checkpoint",
            paths.code_asset(initialization), available,
        )
        for key in ("model_factory", "trainer_factory", "output_adapter"):
            value = specific.get(key)
            checks.append(
                Check(
                    "MODEL",
                    key,
                    "PASS" if available and not _is_placeholder(value) else "FAIL",
                    str(value or "not configured"),
                )
            )
        checks.append(
            Check(
                "DATA",
                "test_labels_not_used_for_tuning",
                "PASS" if specific.get("tuning_splits") == ["train", "validation"] else "FAIL",
                f"tuning_splits={specific.get('tuning_splits')}",
            )
        )
    elif stage == "segmentation":
        import shutil as _shutil

        segmentation = dict(config.get("segmentation") or {})
        backend = str(segmentation.get("backend") or "totalsegmentator")
        if backend == "totalsegmentator":
            executable = str(segmentation.get("executable") or "TotalSegmentator")
            resolved = _shutil.which(executable)
            repository = paths.code_asset(segmentation.get("repository"))
            weights = paths.code_asset(segmentation.get("weights_directory"))
            _exists(checks, "MODEL", "totalsegmentator_repository", repository, True)
            _exists(checks, "MODEL", "totalsegmentator_weights", weights, True)
            checks.append(
                Check(
                    "MODEL",
                    "totalsegmentator_cli",
                    "PASS" if resolved else "FAIL",
                    resolved or executable,
                )
            )
            lungmask = dict(segmentation.get("lungmask") or {})
            if lungmask.get("enabled", True):
                lungmask_cli = str(lungmask.get("executable") or "lungmask")
                lungmask_resolved = _shutil.which(lungmask_cli)
                checkpoint = paths.code_asset(lungmask.get("checkpoint"))
                _exists(checks, "MODEL", "lungmask_checkpoint", checkpoint, True)
                checks.append(
                    Check(
                        "MODEL",
                        "lungmask_cli",
                        "PASS" if lungmask_resolved else "FAIL",
                        lungmask_resolved or lungmask_cli,
                    )
                )
        else:
            checks.append(Check("MODEL", "segmentation_backend", "FAIL", backend))
    elif stage == "roi":
        roi = dict(config.get("roi") or {})
        segmentation_run = paths.output_asset(roi.get("segmentation_run"))
        segmentation_manifest = segmentation_run / "manifest.parquet" if segmentation_run else None
        segmentation_result = segmentation_run / "result.json" if segmentation_run else None
        _exists(checks, "SUPERVISION", "segmentation_run", segmentation_run, True)
        _exists(checks, "SUPERVISION", "segmentation_manifest", segmentation_manifest, True)
        _exists(checks, "SUPERVISION", "segmentation_result", segmentation_result, True)
        if segmentation_result and segmentation_result.is_file():
            try:
                payload = json.loads(segmentation_result.read_text(encoding="utf-8"))
                experiment = dict(payload.get("experiment") or {})
                valid_stage = experiment.get("stage") == "segmentation"
                completed = str(experiment.get("status", "")).startswith("completed")
                valid = valid_stage and completed
                detail = f"stage={experiment.get('stage')} status={experiment.get('status')}"
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                valid = False
                detail = f"invalid result.json: {exc}"
            checks.append(
                Check(
                    "SUPERVISION",
                    "completed_segmentation_run",
                    "PASS" if valid else "FAIL",
                    detail,
                )
            )
    elif stage == "silver":
        silver = dict(config.get("silver") or {})
        method = str(silver.get("method") or "").upper()
        required_roles = (
            ("medgemma",)
            if method == "SL00"
            else ("falcon",)
            if method == "SL01"
            else ("falcon", "medgemma")
        )
        for role in required_roles:
            configured = dict(silver.get(role) or {})
            model_path = paths.code_asset(configured.get("model_path"))
            model_id = configured.get("model_id")
            factory = configured.get("provider_factory")
            reference_ok = model_path.is_dir() if model_path is not None else bool(model_id)
            ok = bool(reference_ok and factory)
            detail = str(model_path or model_id or "not configured")
            checks.append(Check("MODEL", role, "PASS" if ok else "FAIL", detail))
            provider_label = configured.get("provider")
            checks.append(
                Check(
                    "MODEL",
                    f"{role}_provider_label",
                    "PASS" if str(provider_label or "").strip() else "FAIL",
                    str(provider_label or "not configured"),
                )
            )
    lineage = dict(config.get("lineage") or {})
    if lineage.get("source_checkpoint"):
        source_checkpoint_path = Path(str(lineage["source_checkpoint"]))
        _exists(checks, "MODEL", "source_checkpoint", source_checkpoint_path, True)
        if source_checkpoint_path.is_file():
            try:
                source_checkpoint_payload = inspect_checkpoint(source_checkpoint_path)
                lineage_errors = checkpoint_lineage_errors(
                    config, source_checkpoint_payload["lineage"], manifest_rows
                )
                if lineage_errors:
                    for message in lineage_errors:
                        checks.append(
                            Check("LINEAGE", "source_checkpoint_compatibility", "FAIL", message)
                        )
                else:
                    checks.append(
                        Check(
                            "LINEAGE",
                            "source_checkpoint_compatibility",
                            "PASS",
                            f"source_experiment={source_checkpoint_payload['lineage'].get('experiment_id')}",
                        )
                    )
            except CheckpointError as exc:
                checks.append(
                    Check("LINEAGE", "source_checkpoint_compatibility", "FAIL", str(exc))
                )
    supervision = dict(config.get("supervision") or {})
    for key in ("masks", "rois", "silver_labels", "ehr", "pesi"):
        requested = bool(supervision.get(f"require_{key}", False))
        configured = supervision.get(key)
        if stage == "silver_encoder_adaptation" and key == "silver_labels":
            silver_training = dict(config.get("silver_training") or {})
            source = str(silver_training.get("silver_source") or "").upper()
            configured = dict(silver_training.get("sources") or {}).get(source)
            configured = paths.output_asset(configured) if configured else None
            requested = True
        _exists(checks, "SUPERVISION", key, Path(str(configured)) if configured else None, requested)
        if key == "silver_labels" and requested and configured and Path(str(configured)).is_file():
            try:
                from source.silver.schema import TARGETS, canonical_target

                configured_targets = (
                    tuple((config.get("silver_training") or {}).get("targets") or ())
                    if stage == "silver_encoder_adaptation"
                    else tuple(task.get("silver_targets") or ())
                )
                requested_targets = {
                    canonical_target(str(name)) for name in configured_targets
                }
                silver_audit = audit_silver_table(configured, allowed_targets=TARGETS)
                silver_rows = read_rows(configured)
                accepted_keys = [
                    (
                        str(row.get("patient_id") or ""),
                        str(row.get("study_id") or ""),
                        canonical_target(str(row.get("target") or "")),
                    )
                    for row in silver_rows
                    if str(row.get("status") or "") == "accepted"
                ]
                available_targets = {key[2] for key in accepted_keys}
                missing_accepted = sorted(requested_targets - available_targets)
                duplicate_accepted = [
                    key for key, count in Counter(accepted_keys).items()
                    if count > 1 and key[2] in requested_targets
                ]
                silver_errors = list(silver_audit.errors)
                if missing_accepted:
                    silver_errors.append(
                        "configured targets have no accepted rows: "
                        + ", ".join(missing_accepted)
                    )
                if duplicate_accepted:
                    silver_errors.append(
                        "multiple accepted reports map to one requested study-target: "
                        + ", ".join(
                            f"{patient}/{study}/{target}"
                            for patient, study, target in duplicate_accepted[:10]
                        )
                    )
                checks.append(
                    Check(
                        "SUPERVISION", "silver_label_schema",
                        "PASS" if not silver_errors else "FAIL",
                        "; ".join(silver_errors) or f"rows={silver_audit.rows}",
                    )
                )
            except (ManifestError, ValueError) as exc:
                checks.append(Check("SUPERVISION", "silver_label_schema", "FAIL", str(exc)))
    roi_mask_ids = dict(data_config.get("roi_mask_ids") or {})
    if roi_mask_ids:
        raw_roi_manifest = data_config.get("roi_manifest")
        roi_manifest = Path(str(raw_roi_manifest)) if raw_roi_manifest else None
        if roi_manifest is not None and not roi_manifest.is_absolute():
            roi_manifest = paths.output_asset(roi_manifest)
        _exists(checks, "SUPERVISION", "roi_manifest", roi_manifest, True)
        errors: list[str] = []
        if roi_manifest is not None and roi_manifest.is_file() and manifest_rows is not None:
            try:
                roi_rows = read_rows(roi_manifest)
                columns = {key for row in roi_rows for key in row}
                required = {"patient_id", "study_id", "roi_id", "roi_path", "status", "control_for"}
                if missing := sorted(required - columns):
                    errors.append("ROI manifest missing columns: " + ", ".join(missing))
                for region, roi_id in roi_mask_ids.items():
                    expected_control = str(
                        (data_config.get("roi_control_for") or {}).get(region) or ""
                    )
                    found: set[tuple[str, str]] = set()
                    for row in roi_rows:
                        observed_control = str(row.get("control_for") or "")
                        if observed_control.lower() == "nan":
                            observed_control = ""
                        if (
                            str(row.get("roi_id")) == str(roi_id)
                            and observed_control == expected_control
                            and str(row.get("status")) in {"PASS", "SUSPICIOUS"}
                            and Path(str(row.get("roi_path") or "")).is_file()
                        ):
                            found.add((str(row.get("patient_id")), str(row.get("study_id"))))
                    required_studies = {
                        (str(row.get("patient_id")), str(row.get("study_id")))
                        for row in manifest_rows
                    }
                    missing_count = len(required_studies - found)
                    if missing_count:
                        errors.append(
                            f"{region}={roi_id}/{expected_control or 'default'} missing for "
                            f"{missing_count} manifest studies"
                        )
            except (ManifestError, OSError, ValueError) as exc:
                errors.append(str(exc))
        checks.append(
            Check(
                "SUPERVISION", "roi_mask_contract", "PASS" if not errors else "FAIL",
                "; ".join(errors) or f"masks={roi_mask_ids}",
            )
        )
    if effective_stage == "prognosis":
        modalities = set(task.get("modalities") or ())
        if "ehr" in modalities:
            count = len(data_config.get("ehr_columns") or ())
            expected = int(task.get("ehr_input_dim") or 0)
            checks.append(
                Check(
                    "DATA",
                    "ehr_feature_contract",
                    "PASS" if count == expected and count > 0 else "FAIL",
                    f"manifest_columns={count} encoder_input_dim={expected}",
                )
            )
        evaluation = dict(config.get("evaluation") or {})
        split_strategy = str(evaluation.get("split_strategy", "patient_holdout"))
        if split_strategy == "temporal_holdout":
            temporal = dict(evaluation.get("temporal_holdout") or {})
            date_column = str(temporal.get("acquisition_date_column") or "")
            if manifest_rows and date_column:
                from source.data.manifests import audit_temporal_holdout

                audit = audit_temporal_holdout(
                    manifest_rows,
                    date_column=date_column,
                    target_column=str(task.get("primary_target", "mortality_30d")),
                    minimum_events_per_split=int(temporal.get("minimum_events_per_split", 1)),
                    patient_column=str(data_config.get("patient_id_column", "patient_id")),
                    split_column=str(data_config.get("split_column", "split")),
                )
                checks.append(
                    Check(
                        "DATA", "temporal_holdout", "PASS" if audit["ok"] else "FAIL",
                        json.dumps(audit, sort_keys=True),
                    )
                )
            else:
                checks.append(
                    Check(
                        "DATA", "temporal_holdout", "FAIL",
                        "reliable acquisition_date_column and readable manifest are required",
                    )
                )
        elif split_strategy not in {"patient_holdout", "patient_stratified_cv"}:
            checks.append(Check("DATA", "split_strategy", "FAIL", split_strategy))
        if "pesi" in modalities:
            count = len(data_config.get("pesi_columns") or ())
            expected = int(task.get("pesi_input_dim") or 0)
            checks.append(
                Check(
                    "DATA",
                    "pesi_feature_contract",
                    "PASS" if count == expected and count > 0 else "FAIL",
                    f"manifest_columns={count} encoder_input_dim={expected}",
                )
            )
        ehr_full = list(dict.fromkeys(data_config.get("ehr_columns_full") or ()))
        ehr_common = list(dict.fromkeys(data_config.get("ehr_columns_common") or ()))
        if ehr_full or ehr_common:
            subset_ok = set(ehr_common) <= set(ehr_full)
            checks.append(
                Check(
                    "DATA",
                    "ehr_common_subset_of_full",
                    "PASS" if subset_ok else "FAIL",
                    f"full={len(ehr_full)} common={len(ehr_common)} "
                    f"extra_in_common={sorted(set(ehr_common) - set(ehr_full))}",
                )
            )
    if effective_stage == "diagnosis" and task.get("auxiliary_targets"):
        try:
            from source.silver.schema import canonical_target
            from source.tasks.diagnosis.organ_targets import validate_organ_target_supervision

            silver_targets = {
                canonical_target(str(name)) for name in (task.get("silver_targets") or ())
            }
            errors = validate_organ_target_supervision(
                task.get("auxiliary_targets"),
                native_targets=set(data_config.get("label_columns") or ()),
                silver_targets=silver_targets,
                expert_targets=set(data_config.get("expert_label_columns") or ()),
                default_source=str(task.get("auxiliary_default_source", "native")),
            )
            expert_targets = set(data_config.get("expert_label_columns") or ())
            label_targets = set(data_config.get("label_columns") or ())
            if not expert_targets <= label_targets:
                errors.append(
                    "data.expert_label_columns must also be present in data.label_columns: "
                    + str(sorted(expert_targets - label_targets))
                )
            checks.append(
                Check(
                    "SUPERVISION",
                    "organ_auxiliary_target_mapping",
                    "PASS" if not errors else "FAIL",
                    "; ".join(errors) or "all configured targets have anatomically mapped supervision",
                )
            )
        except ValueError as exc:
            checks.append(Check("SUPERVISION", "organ_auxiliary_target_mapping", "FAIL", str(exc)))
    if stage == "counterfactual":
        specification = str(task.get("input_counterfactual") or "")
        region = specification.removeprefix("remove_") if specification.startswith("remove_") else ""
        mask_columns = dict(data_config.get("mask_columns") or {})
        valid = region in {"heart", "pa", "lung", "random"}
        checks.append(
            Check(
                "EXPERIMENT",
                "frozen_counterfactual_contract",
                "PASS" if valid and bool(lineage.get("source_checkpoint")) else "FAIL",
                f"operation={specification} source_checkpoint={lineage.get('source_checkpoint')}",
            )
        )
        roi_masks = set((data_config.get("roi_mask_ids") or {}).keys())
        if region == "random" and "random" in roi_masks | set(mask_columns):
            required_masks = {"random"}
        elif region == "random":
            required_masks = {str(task.get("matched_region", "pa")), "body"}
        else:
            required_masks = {region}
        missing_masks = sorted(required_masks - (set(mask_columns) | roi_masks))
        checks.append(
            Check(
                "DATA",
                "counterfactual_original_masks",
                "PASS" if not missing_masks else "FAIL",
                "original masks reused; missing=" + str(missing_masks),
            )
        )
    if stage == "roi_student":
        specification = str(task.get("input_counterfactual") or "")
        valid = specification in {
            "keep_only_heart", "keep_only_pa", "keep_only_lung", "keep_only_random"
        }
        c0 = str(lineage.get("initialization") or "") == "C0"
        checks.append(
            Check(
                "EXPERIMENT",
                "roi_student_contract",
                "PASS" if valid and c0 else "FAIL",
                f"input={specification} initialization={lineage.get('initialization')}",
            )
        )
        distillation = dict(config.get("distillation") or {})
        if distillation.get("enabled"):
            teacher_checkpoint = Path(str(distillation.get("teacher_checkpoint") or ""))
            _exists(checks, "MODEL", "frozen_teacher_checkpoint", teacher_checkpoint, True)
            values_ok = (
                float(distillation.get("temperature", 0)) > 0
                and float(distillation.get("alpha_distill", -1)) >= 0
                and float(distillation.get("alpha_supervised", -1)) >= 0
            )
            checks.append(
                Check(
                    "EXPERIMENT", "distillation_contract", "PASS" if values_ok else "FAIL",
                    f"temperature={distillation.get('temperature')} "
                    f"lambda_KD={distillation.get('alpha_distill')}",
                )
            )
    if stage == "alignment":
        count = len((config.get("alignment") or {}).get("report_embedding_columns") or ())
        checks.append(
            Check(
                "DATA",
                "report_embedding_contract",
                "PASS" if count > 0 else "FAIL",
                f"columns={count}",
            )
        )
    zero_shot = dict(config.get("zero_shot") or {})
    if zero_shot.get("requested"):
        # A zero-shot request must never fall through to a different stage entrypoint:
        # foundation materialization loads and profiles an encoder, it does not classify.
        checks.append(
            Check(
                "EXPERIMENT",
                "zero_shot_contract",
                "PASS" if zero_shot.get("available") else "FAIL",
                str(
                    zero_shot.get("reason")
                    or "UNAVAILABLE: zero-shot inference is not implemented for this backbone"
                ),
            )
        )
    concept_bottleneck = dict(config.get("concept_bottleneck") or {})
    if bool(concept_bottleneck.get("enabled")):
        try:
            from source.concepts.schema import enabled_concept_specs

            concepts = enabled_concept_specs(concept_bottleneck)
            missing_from_labels = sorted(
                concept.target
                for concept in concepts.values()
                if concept.target not in tuple(data_config.get("label_columns") or ())
            )
            checks.append(
                Check(
                    "SUPERVISION",
                    "concept_bottleneck_targets",
                    "PASS" if not missing_from_labels else "FAIL",
                    f"concepts={sorted(concepts)} missing_from_label_columns={missing_from_labels}",
                )
            )
        except ValueError as exc:
            checks.append(Check("SUPERVISION", "concept_bottleneck_targets", "FAIL", str(exc)))
    if stage == "silver_encoder_adaptation":
        silver_training = dict(config.get("silver_training") or {})
        targets = silver_training.get("targets") or {}
        source = str(silver_training.get("silver_source") or "")
        required = bool(supervision.get("require_silver_labels", False))
        valid_targets = isinstance(targets, Mapping) and bool(targets) and all(
            int(classes) > 0 for classes in targets.values()
        )
        checks.append(
            Check(
                "SUPERVISION",
                "silver_encoder_adaptation_contract",
                "PASS" if valid_targets and required and source in {"SL00", "SL01", "SL02"} else "FAIL",
                f"source={source} require_silver_labels={required} targets={len(targets)}",
            )
        )
        initialization = dict(config.get("init") or {})
        checkpoint = paths.output_asset(initialization.get("checkpoint"))
        _exists(checks, "MODEL", "C0_checkpoint", checkpoint, True)
    compute = dict(config.get("compute") or {})
    strategy = str(compute.get("strategy", "single"))
    accelerator = str(compute.get("accelerator", "cuda" if strategy != "cpu" else "cpu"))
    devices = list(compute.get("devices") or [])
    if accelerator == "cuda" or devices:
        try:
            import torch

            detected = torch.cuda.device_count() if torch.cuda.is_available() else 0
            valid = detected > 0 and all(0 <= int(device) < detected for device in devices)
            checks.append(
                Check(
                    "COMPUTE",
                    "cuda_devices",
                    "PASS" if valid else "FAIL",
                    f"requested={devices} detected={detected}",
                )
            )
        except ModuleNotFoundError:
            checks.append(Check("COMPUTE", "cuda_devices", "FAIL", "PyTorch is not installed"))
    else:
        checks.append(Check("COMPUTE", "cpu", "PASS", "CPU engineering mode"))
    if str((config.get("experiment") or {}).get("stage") or "") == "dataset":
        # A dataset build writes to the derived data root, not to the experiment output
        # root, so the output-family collision rule does not apply to it. What matters is
        # whether this profile has already been built.
        try:
            destination = paths.dataset_root_for(config)
        except PathConfigurationError as exc:
            checks.append(Check("OUTPUT", "dataset_destination", "FAIL", str(exc)))
        else:
            built = (destination / "dataset.json").is_file()
            rebuild_ok = not built or bool(config.get("overwrite"))
            checks.append(
                Check(
                    "OUTPUT", "dataset_destination", "PASS" if rebuild_ok else "FAIL",
                    f"{destination} ({'already built; pass --overwrite to rebuild' if built else 'clear'})",
                )
            )
    elif paths.output_root is not None:
        experiment = dict(config.get("experiment") or {})
        family = str(experiment.get("family") or experiment.get("stage"))
        experiment_id = str((config.get("experiment") or {}).get("id") or "")
        try:
            manager = OutputManager(paths)
            state = manager.inspect_collision(family, experiment_id)
            collision_ok = state is None or bool(config.get("resume")) or bool(config.get("overwrite"))
            checks.append(Check("OUTPUT", "collision", "PASS" if collision_ok else "FAIL", state or "clear"))
        except (KeyError, ValueError, PathConfigurationError) as exc:
            checks.append(Check("OUTPUT", "collision", "FAIL", str(exc)))
    return PreflightReport(tuple(checks), manifest_payload, source_checkpoint_payload)


def require_preflight(config: Mapping[str, Any], paths: ProjectPaths) -> PreflightReport:
    report = run_preflight(config, paths)
    if not report.ok:
        failures = [
            f"{item.group}/{item.name}: {item.detail}"
            for item in report.checks
            if item.status == "FAIL"
        ]
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(failures))
    return report
