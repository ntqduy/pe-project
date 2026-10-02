from __future__ import annotations

import importlib.util
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
from source.engine.experiment import OutputManager, run_output_id, verify_writable_directory


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
        from source.data.profiles import ACTIVE_PROFILES
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
    if effective_stage in {"ablation", "counterfactual"}:
        effective_stage = str(task.get("base_stage") or "")
    source_dataset = str(source_lineage.get("dataset") or "").lower()
    source_stage = str(source_lineage.get("stage") or "").lower()
    source_task = str(source_lineage.get("task") or "").lower()
    source_supervision = str(source_lineage.get("supervision_type") or "").lower()
    inspect_supervised_diagnosis = (
        _is_inspect_dataset(source_dataset)
        and (source_stage == "diagnosis" or "diagnos" in source_task)
        and source_supervision not in {"none", "public", "self_supervised"}
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
        if stage in {"ablation", "counterfactual"}
        else stage
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
            from source.data.profiles import assert_shared_preprocessing, require_active_profile

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
            if release is not None:
                for module in ("numpy", "nibabel", "pyarrow"):
                    present = importlib.util.find_spec(module) is not None
                    checks.append(
                        Check(
                            "DEPENDENCY",
                            module,
                            "PASS" if present else "FAIL",
                            "installed" if present else "install requirements.txt before building the dataset",
                        )
                    )
                ehr_config = dict(profile.get("ehr") or {})
                if bool(ehr_config.get("enabled", True)):
                    archive = release / "EHR" / str(ehr_config.get("archive") or "")
                    _exists(checks, "DATA", "ehr_archive", archive, True)
                    policy = ehr_config.get("prohibited")
                    valid_policy = isinstance(policy, dict) and isinstance(
                        policy.get("excluded_tables"), list
                    ) and isinstance(policy.get("code_description_patterns"), list)
                    checks.append(
                        Check(
                            "DATA",
                            "ehr_prohibited_policy",
                            "PASS" if valid_policy else "FAIL",
                            "inline leakage policy configured"
                            if valid_policy
                            else (
                                "ehr.prohibited must define excluded_tables and "
                                "code_description_patterns lists"
                            ),
                        )
                    )
                spesi_config = dict(profile.get("spesi") or {})
                if bool(spesi_config.get("enabled", True)):
                    mapping = Path(str(spesi_config.get("mapping_config") or ""))
                    if not mapping.is_absolute():
                        mapping = paths.code_root / mapping
                    _exists(checks, "DATA", "spesi_mapping_contract", mapping, True)
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
                file_column=data_config.get("file_column", "image_path"),
                data_root=paths.dataset_root_for(config),
                split_aliases=data_config.get("split_aliases"),
                required_columns=tuple(
                    dict.fromkeys(
                        (
                            data_config.get("file_column", "image_path"),
                            *tuple(data_config.get("label_columns") or ()),
                            *tuple(data_config.get("ehr_columns") or ()),
                            *tuple(data_config.get("spesi_columns") or ()),
                            *tuple(
                                value
                                for value in (
                                    data_config.get("ehr_availability_column"),
                                    data_config.get("spesi_availability_column"),
                                )
                                if value
                            ),
                            *tuple((data_config.get("mask_columns") or {}).values()),
                        )
                    )
                ),
                # Diagnosis is a fully observed classification target, so every row must
                # carry its primary label. Prognosis endpoints are time-to-event style
                # outcomes: rows can legitimately be censored/unobserved and are masked by
                # CTPADataset + masked_multitask_loss. The label columns themselves remain
                # required above; only the per-row primary-label requirement is relaxed for
                # prognosis. A later split-level check below ensures each split still has
                # usable supervision.
                primary_target=(
                    str(task.get("primary_target"))
                    if effective_stage == "diagnosis" and task.get("primary_target")
                    else None
                ),
                fold_column=data_config.get("fold_column"),
                required_splits=(
                    ("test",)
                    if bool((config.get("external_evaluation") or {}).get("test_only"))
                    else ("train", "validation", "test")
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
            if (
                effective_stage == "prognosis"
                and manifest_rows is not None
                and task.get("primary_target")
            ):
                target = str(task["primary_target"])
                split_column = str(data_config.get("split_column", "split"))
                valid_by_split: dict[str, int] = {}
                for row in manifest_rows:
                    split = str(row.get(split_column) or "").strip().lower()
                    value = row.get(target)
                    if split and value is not None and str(value).strip() != "":
                        valid_by_split[split] = valid_by_split.get(split, 0) + 1
                missing_splits = [
                    split for split in ("train", "validation", "test")
                    if valid_by_split.get(split, 0) == 0
                ]
                checks.append(
                    Check(
                        "DATA",
                        "prognosis_primary_target_coverage",
                        "FAIL" if missing_splits else "PASS",
                        (
                            f"target={target} valid_by_split={dict(sorted(valid_by_split.items()))}; "
                            f"no usable labels in: {', '.join(missing_splits)}"
                            if missing_splits
                            else f"target={target} valid_by_split={dict(sorted(valid_by_split.items()))}"
                        ),
                    )
                )
        except (ManifestError, PathConfigurationError) as exc:
            checks.append(Check("DATA", "split_manifest", "FAIL", str(exc)))
    model = dict(config.get("model") or {})
    backbone = str(model.get("backbone") or "")
    if stage in {"diagnosis", "prognosis", "ablation", "counterfactual"}:
        uses_image_model = not (
            effective_stage == "prognosis"
            and "image" not in set(task.get("modalities") or ())
        )
        if not uses_image_model:
            checks.append(Check("MODEL", "image_backbone", "SKIP", "image modality disabled"))
        elif backbone and str(model.get("integration") or "external") == "baseline":
            # 2D / 2.5D / 3D baseline zoo (source/model). Weights resolve at build time and the
            # training log states whether they loaded; only a pinned local file is checked here.
            from source.components.encoders.image.registry import registered_backbones

            registered = backbone.strip().lower().replace("-", "_") in registered_backbones()
            checks.append(
                Check(
                    "MODEL",
                    f"{backbone}_baseline_contract",
                    "PASS" if registered and int(model.get("feature_dim") or 0) > 0 else "FAIL",
                    "registered baseline encoder (source/model/registry.py) with feature_dim",
                )
            )
            pretrained = dict(model.get("pretrained") or {})
            wants_weights = bool(model.get("load_pretrained", True)) and bool(pretrained.get("enabled", True))
            if wants_weights and pretrained.get("path"):
                present = paths.code_asset(pretrained["path"]).is_file()
                downloadable = bool(pretrained.get("url")) and bool(pretrained.get("allow_download", True))
                checks.append(
                    Check(
                        "MODEL",
                        f"{backbone}_pretrained_file",
                        # SKIP is the non-blocking status: preflight only fails on FAIL.
                        "PASS" if present else ("SKIP" if downloadable or not pretrained.get("required") else "FAIL"),
                        str(paths.code_asset(pretrained["path"]))
                        + ("" if present else " missing (downloaded at build time)" if downloadable
                           else " missing -> training from scratch" if not pretrained.get("required") else " missing"),
                    )
                )
            head_type = str((config.get("head") or {}).get("type") or "")
            checks.append(
                Check("MODEL", "classification_head", "PASS" if head_type in {"mlp", "kan"} else "FAIL",
                      f"head.type={head_type or 'missing'}")
            )
        elif backbone and str(model.get("integration") or "external") == "local":
            from source.components.encoders.image.registry import registered_backbones

            contract_ok = (
                backbone.strip().lower().replace("-", "_") in registered_backbones()
                and int(model.get("feature_dim") or 0) > 0
                and not bool(model.get("load_pretrained", False))
            )
            checks.append(
                Check(
                    "MODEL",
                    f"{backbone}_local_contract",
                    "PASS" if contract_ok else "FAIL",
                    "registered project-native encoder; random initialization",
                )
            )
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
                completed = experiment.get("status") == "completed"
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
        from source.silver.generator import SILVER_METHODS

        silver = dict(config.get("silver") or {})
        method = str(silver.get("method") or "").strip().lower()
        # The method name is the cascade, so it also decides which providers must be
        # staged. Reading it from SILVER_METHODS keeps this in step with the generator
        # instead of repeating the mapping here.
        stages = SILVER_METHODS.get(method)
        if stages is None:
            checks.append(
                Check(
                    "SILVER",
                    "method",
                    "FAIL",
                    f"silver.method={method!r} is not one of {sorted(SILVER_METHODS)}",
                )
            )
        required_roles = tuple(
            name for name in (stages or ()) if name in {"medgemma"}
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
    # masks / rois / silver_labels are produced by earlier stages under the output root;
    # ehr / spesi are stage-0 artifacts inside the dataset profile. Relative values resolve
    # against the right root so no supervision path has to be hard-coded.
    supervision_roots = {
        "masks": True, "rois": True, "silver_labels": True, "ehr": False, "spesi": False,
    }
    for key, in_output in supervision_roots.items():
        requested = bool(supervision.get(f"require_{key}", False))
        configured = supervision.get(key)
        try:
            configured = _artifact_path(
                paths, configured, output=in_output, profile=data_config.get("profile")
            )
        except PathConfigurationError as exc:
            checks.append(Check("SUPERVISION", key, "FAIL", str(exc)))
            continue
        _exists(checks, "SUPERVISION", key, configured, requested)
        if key == "silver_labels" and requested and configured and Path(str(configured)).is_file():
            try:
                from source.silver.schema import TARGETS, canonical_target

                configured_targets = tuple(task.get("silver_targets") or ())
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
                            and str(row.get("status")) in {"PASS", "SUSPICIOUS", "pass"}
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
            ehr_profile = str(data_config.get("ehr_profile") or "").strip()
            profiles_path = paths.dataset_cache_root(str(data_config.get("profile") or paths.dataset_root_for(config).name)) / "clinical" / "ehr_profiles.json"
            if not profiles_path.is_file():
                profiles_path = paths.dataset_root_for(config) / "clinical" / "ehr_profiles.json"
            if not ehr_profile:
                checks.append(
                    Check(
                        "DATA",
                        "ehr_temporal_profile",
                        "FAIL",
                        "data.ehr_profile is required for an EHR prognosis arm",
                    )
                )
            else:
                try:
                    profile_payload = json.loads(profiles_path.read_text(encoding="utf-8"))
                    profiles = dict(profile_payload.get("profiles") or {})
                    selected = dict(profiles.get(ehr_profile) or {})
                    configured_columns = tuple(data_config.get("ehr_columns") or ())
                    recorded_columns = tuple(selected.get("manifest_feature_columns") or ())
                    cutoff = selected.get("end_before_ctpa_hours")
                    operator = str(selected.get("feature_end_operator") or "")
                    expected_cutoff = {"EHR_0_h": 0.0, "EHR_24_h": 24.0}.get(ehr_profile)
                    profile_ok = bool(selected) and cutoff is not None and float(cutoff) >= 0
                    profile_ok = profile_ok and operator == (
                        "event_time < procedure_datetime - end_before_ctpa_hours"
                    )
                    if expected_cutoff is not None:
                        profile_ok = profile_ok and float(cutoff) == expected_cutoff
                    profile_ok = profile_ok and configured_columns == recorded_columns
                    checks.append(
                        Check(
                            "DATA",
                            "ehr_temporal_profile",
                            "PASS" if profile_ok else "FAIL",
                            (
                                f"profile={ehr_profile} cutoff_hours={cutoff} "
                                f"operator={operator!r} columns={len(recorded_columns)}"
                            ),
                        )
                    )
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    checks.append(
                        Check(
                            "DATA",
                            "ehr_temporal_profile",
                            "FAIL",
                            f"{profiles_path}: {exc}",
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
                    target_column=str(task.get("primary_target") or "1_month_mortality"),
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
        elif split_strategy != "patient_holdout":
            checks.append(Check("DATA", "split_strategy", "FAIL", split_strategy))
        if "spesi" in modalities:
            count = len(data_config.get("spesi_columns") or ())
            expected = int(task.get("spesi_input_dim") or 0)
            checks.append(
                Check(
                    "DATA",
                    "spesi_feature_contract",
                    "PASS" if count == expected and count > 0 else "FAIL",
                    f"manifest_columns={count} encoder_input_dim={expected}",
                )
            )
            status_path = paths.dataset_cache_root(str(data_config.get("profile") or paths.dataset_root_for(config).name)) / "clinical" / "spesi_status.json"
            if not status_path.is_file():
                status_path = paths.dataset_root_for(config) / "clinical" / "spesi_status.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                checks.append(Check("DATA", "spesi_readiness", "FAIL", f"{status_path}: {exc}"))
            else:
                scored = int(status.get("scored_cases") or 0)
                by_split = dict(status.get("scored_cases_by_split") or {})
                covered = all(int(by_split.get(split) or 0) > 0 for split in ("train", "validation", "test"))
                checks.append(
                    Check(
                        "DATA",
                        "spesi_readiness",
                        "PASS" if status.get("status") == "built" and covered else "FAIL",
                        f"status={status.get('status', 'missing')}; scored={scored}; "
                        f"by_split={dict(sorted(by_split.items()))}; reason={status.get('reason', '')}",
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
        # A trained task run is keyed per epoch budget (<id>/epoch_<E>), so only repeating
        # the same training.epochs collides; other budgets of the same experiment coexist.
        output_id = run_output_id(config)
        try:
            manager = OutputManager(paths)
            state = manager.inspect_collision(family, output_id)
            collision_ok = state is None or bool(config.get("resume")) or bool(config.get("overwrite"))
            detail = "clear" if state is None else f"{state}: {manager.run_dir(family, output_id)}"
            if not collision_ok:
                detail += " already exists; pass --overwrite to replace only this run"
            checks.append(Check("OUTPUT", "collision", "PASS" if collision_ok else "FAIL", detail))
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
