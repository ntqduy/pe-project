"""PESI/sPESI readiness gate and score materialisation.

INSPECT distributes raw MEDS/OMOP events rather than a validated PESI table.  This
module therefore has two deliberately separate behaviours:

* it always writes a de-identified mapping audit against ``codes.parquet``; and
* it writes PESI/sPESI features *only* after an approved component contract and a
  supplied, study-level component table are present.

The component table bridge is intentional.  Choosing cancer, heart-failure, chronic
lung disease, altered-mental-status phenotypes, vital units, and an index-time window
is clinical data stewardship work.  Pattern matching raw EHR descriptions would make a
plausible-looking but scientifically invalid score.
"""
from __future__ import annotations

import csv
import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from source.clinical.pesi import DEFAULT_FIELDS, compute_pesi

from .ehr import _as_patient_id, _load_code_descriptions, _load_pyarrow, _parse_time
from .sources import StudyRecord


class PesiBuildError(RuntimeError):
    pass


PESI_COMPONENTS = tuple(DEFAULT_FIELDS)
PESI_FEATURE_COLUMNS = (
    "pesi",
    "spesi",
    "pesi_class",
    "pesi_computable",
    "spesi_computable",
)


@dataclass(frozen=True)
class PesiArtifacts:
    """PESI provenance plus feature columns safe to merge into manifests."""

    status: str
    feature_columns: tuple[str, ...]
    by_study: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency environment specific
        raise PesiBuildError("PESI mapping requires PyYAML; install requirements.txt") from exc
    if not path.is_file():
        raise PesiBuildError(f"PESI mapping configuration is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - YAML parser error types are version dependent
        raise PesiBuildError(f"invalid PESI mapping configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PesiBuildError(f"PESI mapping must contain a mapping: {path}")
    return payload


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        raise PesiBuildError(f"PESI source code list must be a string/list, got {type(value).__name__}")
    return [str(item).strip() for item in value if str(item).strip()]


def _resolve_mapping_path(value: Any, code_root: Path) -> Path:
    text = os.path.expandvars(str(value or "").strip())
    if not text or "${" in text:
        raise PesiBuildError("pesi.mapping_config must be a resolved path")
    path = Path(text)
    return path if path.is_absolute() else code_root / path


def _resolve_component_table(value: Any, code_root: Path) -> Path | None:
    text = os.path.expandvars(str(value or "").strip())
    if not text:
        return None
    if "${" in text:
        raise PesiBuildError("pesi.components_table contains an unresolved environment variable")
    path = Path(text)
    return path if path.is_absolute() else code_root / path


def _mapping_audit(
    mapping: Mapping[str, Any], *, mapping_path: Path, code_descriptions: Mapping[str, str]
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    if int(mapping.get("schema_version") or 0) != 1:
        raise PesiBuildError("PESI mapping schema_version must be 1")
    components = mapping.get("components")
    if not isinstance(components, Mapping):
        raise PesiBuildError("PESI mapping must define components")
    missing = [component for component in PESI_COMPONENTS if component not in components]
    extra = sorted(set(components) - set(PESI_COMPONENTS))
    if missing or extra:
        raise PesiBuildError(
            "PESI mapping components must match the score contract; "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )

    input_columns: dict[str, str] = {}
    component_audit: dict[str, Any] = {}
    blockers: list[str] = []
    for component in PESI_COMPONENTS:
        item = components[component]
        if not isinstance(item, Mapping):
            raise PesiBuildError(f"PESI component {component} must be a mapping")
        status = str(item.get("status") or "").strip().lower()
        if status not in {"approved", "candidate", "unmapped"}:
            raise PesiBuildError(
                f"PESI component {component}.status must be approved, candidate or unmapped"
            )
        column = str(item.get("input_column") or "").strip()
        if not column:
            raise PesiBuildError(f"PESI component {component}.input_column is required")
        input_columns[component] = column
        codes = [
            *_as_string_list(item.get("source_codes")),
            *_as_string_list(item.get("positive_codes")),
            *_as_string_list(item.get("negative_codes")),
        ]
        catalog = str(item.get("catalog") or "codes_parquet").strip().lower()
        code_audit = []
        for code in sorted(set(codes)):
            if catalog == "special_meds_event":
                code_audit.append({"code": code, "status": "special_event_not_in_codes_parquet"})
            elif code in code_descriptions:
                code_audit.append(
                    {"code": code, "status": "present", "description": code_descriptions[code]}
                )
            else:
                code_audit.append({"code": code, "status": "not_found"})
                blockers.append(f"{component}: configured code absent from codes.parquet ({code})")
        if status != "approved":
            blockers.append(f"{component}: mapping status is {status}")
        if status == "approved" and not codes:
            blockers.append(f"{component}: approved mapping has no source code")
        component_audit[component] = {
            "status": status,
            "input_column": column,
            "type": str(item.get("type") or ""),
            "source_table": str(item.get("source_table") or ""),
            "selection": str(item.get("selection") or ""),
            "expected_unit": item.get("expected_unit"),
            "required_review": str(item.get("required_review") or ""),
            "codes": code_audit,
        }

    approval = dict(mapping.get("clinical_approval") or {})
    approval_status = str(approval.get("status") or "pending").strip().lower()
    if approval_status != "approved":
        blockers.insert(0, f"clinical approval status is {approval_status}")
    return (
        {
            "mapping_config": str(mapping_path),
            "schema_version": int(mapping["schema_version"]),
            "score_specification": str(mapping.get("score_specification") or ""),
            "clinical_approval": approval,
            "index": dict(mapping.get("index") or {}),
            "components": component_audit,
        },
        blockers,
        input_columns,
    )


def _ehr_source_root(ehr_metadata: Mapping[str, Any]) -> Path:
    extraction = dict(ehr_metadata.get("extraction") or {})
    source_root = Path(str(extraction.get("cache") or ""))
    if not (source_root / "metadata" / "codes.parquet").is_file():
        raise PesiBuildError(
            "cannot audit PESI source codes because EHR extraction metadata does not point to codes.parquet"
        )
    return source_root


def _source_code_descriptions(source_root: Path) -> dict[str, str]:
    codebook = source_root / "metadata" / "codes.parquet"
    _, _, _, parquet = _load_pyarrow()
    return _load_code_descriptions(codebook, parquet)


def _candidate_coverage(
    records: Sequence[StudyRecord],
    *,
    mapping: Mapping[str, Any],
    source_root: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Aggregate candidate PESI-event availability without materialising patient data.

    This is a readiness audit, not a scorer. Candidate events are never written with a
    patient/study ID or value; it reports only code-level counts, units and value ranges
    needed for a clinician to decide whether the mapping is defensible.
    """
    components = dict(mapping.get("components") or {})
    index = dict(mapping.get("index") or {})
    vital_window = timedelta(hours=float(index.get("vital_lookback_hours") or 0))
    code_to_components: dict[str, list[str]] = defaultdict(list)
    component_codes: dict[str, list[str]] = {}
    for component in PESI_COMPONENTS:
        item = dict(components.get(component) or {})
        codes = list(
            dict.fromkeys(
                [
                    *_as_string_list(item.get("source_codes")),
                    *_as_string_list(item.get("positive_codes")),
                    *_as_string_list(item.get("negative_codes")),
                ]
            )
        )
        component_codes[component] = codes
        for code in codes:
            code_to_components[code].append(component)

    by_patient: dict[str, list[StudyRecord]] = defaultdict(list)
    for record in records:
        if record.has_ehr_crosswalk:
            by_patient[record.patient_id].append(record)
    linked_records = [record for records_for_patient in by_patient.values() for record in records_for_patient]
    output: dict[str, dict[str, Any]] = {
        component: {
            "mapping_status": str(dict(components.get(component) or {}).get("status") or ""),
            "candidate_codes": component_codes[component],
            "crosswalk_linked_cases": 0,
            "missing_index_time_cases": 0,
            "events_loaded": 0,
            "events_missing_time": 0,
            "events_at_or_after_index": 0,
            "events_outside_pre_ctpa_window": 0,
            "cases_with_eligible_candidate_event": 0,
            "eligible_numeric_events": 0,
            "units": {},
            "numeric_min": None,
            "numeric_max": None,
        }
        for component in PESI_COMPONENTS
    }
    for component in PESI_COMPONENTS:
        values = output[component]
        values["crosswalk_linked_cases"] = len(linked_records)
        values["missing_index_time_cases"] = sum(
            _parse_time(record.procedure_datetime) is None for record in linked_records
        )
    if not code_to_components or not by_patient:
        return {
            "status": "not_scanned" if not code_to_components else "no_crosswalk_linked_cases",
            "feature_end_operator": "event_time < procedure_datetime",
            "vital_lookback_hours": float(index.get("vital_lookback_hours") or 0),
            "components": output,
        }

    pa, pc, dataset_module, _ = _load_pyarrow()
    data_directory = source_root / "data"
    try:
        dataset = dataset_module.dataset(data_directory, format="parquet")
    except Exception as exc:  # noqa: BLE001 - pyarrow exceptions differ by version
        raise PesiBuildError(f"cannot open MEDS event dataset for PESI audit: {exc}") from exc
    available = set(dataset.schema.names)
    required = {"subject_id", "time", "code"}
    missing = sorted(required - available)
    if missing:
        raise PesiBuildError("MEDS event dataset lacks PESI-audit columns: " + ", ".join(missing))
    patient_ids = sorted({_as_patient_id(patient) for patient in by_patient})
    columns = [name for name in ("subject_id", "time", "code", "numeric_value", "unit") if name in available]
    try:
        filter_expression = (
            pc.is_in(dataset_module.field("subject_id"), value_set=pa.array(patient_ids))
            & pc.is_in(dataset_module.field("code"), value_set=pa.array(sorted(code_to_components)))
        )
        scanner = dataset.scanner(columns=columns, filter=filter_expression, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001
        raise PesiBuildError(f"cannot scan MEDS candidate PESI events: {exc}") from exc

    units: dict[str, Counter[str]] = {component: Counter() for component in PESI_COMPONENTS}
    cases_with_event: dict[str, set[str]] = {component: set() for component in PESI_COMPONENTS}
    for batch in scanner.to_batches():
        values = batch.to_pydict()
        for row_index in range(batch.num_rows):
            patient = str(values["subject_id"][row_index])
            matching_cases = by_patient.get(patient) or ()
            if not matching_cases:
                continue
            code = str(values["code"][row_index] or "")
            event_time = _parse_time(values["time"][row_index])
            numeric_value = values.get("numeric_value", [None])[row_index]
            unit = str(values.get("unit", [None])[row_index] or "").strip() or "MISSING"
            for component in code_to_components.get(code, ()):
                detail = output[component]
                is_vital = str(dict(components.get(component) or {}).get("type") or "") == "numeric"
                for record in matching_cases:
                    detail["events_loaded"] += 1
                    index_time = _parse_time(record.procedure_datetime)
                    if index_time is None or event_time is None:
                        detail["events_missing_time"] += 1
                        continue
                    if event_time >= index_time:
                        detail["events_at_or_after_index"] += 1
                        continue
                    if is_vital and event_time < index_time - vital_window:
                        detail["events_outside_pre_ctpa_window"] += 1
                        continue
                    cases_with_event[component].add(record.study_id)
                    units[component][unit] += 1
                    if numeric_value is not None:
                        try:
                            number = float(numeric_value)
                        except (TypeError, ValueError):
                            continue
                        if number == number and number not in {float("inf"), float("-inf")}:
                            detail["eligible_numeric_events"] += 1
                            detail["numeric_min"] = (
                                number if detail["numeric_min"] is None else min(detail["numeric_min"], number)
                            )
                            detail["numeric_max"] = (
                                number if detail["numeric_max"] is None else max(detail["numeric_max"], number)
                            )
    for component, detail in output.items():
        detail["cases_with_eligible_candidate_event"] = len(cases_with_event[component])
        detail["units"] = dict(sorted(units[component].items()))
    return {
        "status": "scanned",
        "feature_end_operator": "event_time < procedure_datetime",
        "vital_lookback_hours": float(index.get("vital_lookback_hours") or 0),
        "components": output,
    }


def _read_component_rows(
    table: Path,
    *,
    selected_studies: set[str],
    input_columns: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    if not table.is_file():
        raise PesiBuildError(f"approved PESI component table is missing: {table}")
    with table.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"study_id", *input_columns.values()}
        missing = sorted(required - fields)
        if missing:
            raise PesiBuildError("PESI component table lacks required columns: " + ", ".join(missing))
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            study_id = str(row.get("study_id") or "").strip()
            if not study_id or study_id not in selected_studies:
                continue
            if study_id in rows:
                raise PesiBuildError(f"PESI component table has duplicate selected study_id={study_id}")
            rows[study_id] = {column: str(row.get(column) or "") for column in input_columns.values()}
    return rows


def _materialize_scores(
    records: Sequence[StudyRecord],
    component_rows: Mapping[str, Mapping[str, str]],
    input_columns: Mapping[str, str],
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_study: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    component_missing = Counter()
    for record in records:
        components = dict(component_rows.get(record.study_id) or {})
        for component, column in input_columns.items():
            if not str(components.get(column) or "").strip():
                component_missing[component] += 1
        try:
            result = compute_pesi(components, fields=input_columns)
        except (TypeError, ValueError) as exc:
            raise PesiBuildError(f"invalid PESI component value for study {record.study_id}: {exc}") from exc
        features = {
            "pesi": result.pesi_score,
            "spesi": result.spesi_score,
            "pesi_class": result.pesi_class,
            "pesi_computable": int(result.pesi_computable),
            "spesi_computable": int(result.spesi_computable),
        }
        by_study[record.study_id] = features
        rows.append(
            {
                "patient_id": record.patient_id,
                "study_id": record.study_id,
                "split": record.split,
                **features,
            }
        )
    path = output_dir / "pesi_features.csv"
    _write_csv(path, rows, ["patient_id", "study_id", "split", *PESI_FEATURE_COLUMNS])
    return by_study, {
        "feature_table": str(path),
        "component_rows_for_selected_studies": len(component_rows),
        "component_missing_cases": dict(sorted(component_missing.items())),
        "pesi_computable_cases": sum(row["pesi_computable"] for row in rows),
        "spesi_computable_cases": sum(row["spesi_computable"] for row in rows),
    }


def build_pesi_artifacts(
    records: Sequence[StudyRecord],
    *,
    output_dir: Path,
    config: Mapping[str, Any] | None,
    ehr_metadata: Mapping[str, Any],
    code_root: Path,
) -> PesiArtifacts:
    """Audit the PESI contract; score only an explicitly approved component table."""
    settings = dict(config or {})
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(settings.get("enabled", True)):
        metadata = {"status": "disabled", "reason": "pesi.enabled=false"}
        _write_json(output_dir / "pesi_status.json", metadata)
        return PesiArtifacts("disabled", (), {}, metadata)
    mapping_path = _resolve_mapping_path(settings.get("mapping_config"), code_root)
    mapping = _read_yaml(mapping_path)
    source_root = _ehr_source_root(ehr_metadata)
    audit, blockers, input_columns = _mapping_audit(
        mapping,
        mapping_path=mapping_path,
        code_descriptions=_source_code_descriptions(source_root),
    )
    candidate_coverage: dict[str, Any] | None = None
    if bool(settings.get("audit_candidate_coverage", True)):
        batch_size = int(settings.get("audit_batch_size", 250000))
        if batch_size < 1:
            raise PesiBuildError("pesi.audit_batch_size must be positive")
        candidate_coverage = _candidate_coverage(
            records,
            mapping=mapping,
            source_root=source_root,
            batch_size=batch_size,
        )
    components_table = _resolve_component_table(settings.get("components_table"), code_root)
    metadata: dict[str, Any] = {
        "status": "blocked",
        "reason": "; ".join(blockers) or "PESI component source table is not configured",
        "required_columns": list(settings.get("required_columns") or ["pesi", "spesi"]),
        "mapping_audit": audit,
        "candidate_coverage": candidate_coverage,
        "components_table": str(components_table) if components_table else None,
        "score_output_columns": list(PESI_FEATURE_COLUMNS),
    }
    # No score is emitted until the clinical contract is fully approved. This keeps
    # prognosis runs blocked instead of allowing an all-missing score through silently.
    if blockers:
        _write_json(output_dir / "pesi_status.json", metadata)
        _write_json(output_dir / "pesi_mapping_audit.json", audit)
        return PesiArtifacts("blocked", (), {}, metadata)
    if components_table is None:
        metadata["reason"] = "approved mapping needs pesi.components_table"
        _write_json(output_dir / "pesi_status.json", metadata)
        _write_json(output_dir / "pesi_mapping_audit.json", audit)
        return PesiArtifacts("blocked", (), {}, metadata)

    selected_studies = {record.study_id for record in records}
    components = _read_component_rows(
        components_table,
        selected_studies=selected_studies,
        input_columns=input_columns,
    )
    by_study, coverage = _materialize_scores(records, components, input_columns, output_dir)
    complete = coverage["pesi_computable_cases"] > 0 and coverage["spesi_computable_cases"] > 0
    metadata.update(coverage)
    metadata["status"] = "built" if complete else "built_no_complete_cases"
    metadata["reason"] = (
        "approved PESI/sPESI component table scored"
        if complete
        else "approved component table was read, but no selected case has all required values"
    )
    _write_json(output_dir / "pesi_status.json", metadata)
    _write_json(output_dir / "pesi_mapping_audit.json", audit)
    return PesiArtifacts(metadata["status"], PESI_FEATURE_COLUMNS, by_study, metadata)
