"""Derived manifest construction for the project's training stages.

Ported from /mnt/pe_study `build_inspect_manifest.py`, `build_inspect_500_phase1_audit.py`
and `build_prognosis_cohort.py`, unified so one cohort pass emits every manifest the
pipeline consumes, all from the same eligible study set.

Emitted under ``<derived>/datasets/<profile>/manifests/``:

    ctpa.csv            every eligible study; the CTPA-level manifest (segmentation, DAPT)
    diagnosis.csv       the diagnosis cohort with native PE labels
    prognosis.csv       legacy confirmed-acute-PE index cohort (30-day mortality alias)
    prognosis_all_patient.csv  one index CTPA per patient, all prognosis outcomes
    prognosis_pe_positive.csv  one PE-positive index CTPA per patient, all outcomes
    paired_reports.csv  image/report pairs for the alignment stage
    reports.csv         report text for silver-label generation

Column names follow the project schema (`patient_id`, `study_id`, `split`, `image_path`),
with the INSPECT identifiers (`person_id`, `image_id`, `impression_id`) carried alongside
so any row can be traced back to the release.
"""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adjudication import first_index_study, mortality_outcome, normalize_binary
from .sources import StudyRecord

# INSPECT column -> project target name. Native labels only; every richer attribute
# (acuity class, clot location, RV findings, ...) is silver-derived and never invented here.
DEFAULT_LABEL_MAP = {
    "pe_positive_nlp": "pe_present",
    "pe_acute": "pe_acute",
    "pe_subsegmentalonly": "pe_subsegmental_only",
}
DEFAULT_DIAGNOSIS_LABEL_ALIASES = {
    # Matrix launchers use the clinical names requested by the current protocol.  Keep
    # the historical pe_present / pe_subsegmental_only columns too so old configs stay
    # byte-for-byte compatible.
    "pe_positive": "pe_positive_nlp",
    "pe_subsegmental": "pe_subsegmentalonly",
}
DEFAULT_PROGNOSIS_LABEL = "1_month_mortality"
DEFAULT_PROGNOSIS_CONDITIONS = {"pe_positive_nlp": "TRUE", "pe_acute": "TRUE"}
DEFAULT_PROGNOSIS_OUTCOMES = (
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
)
DEFAULT_PROGNOSIS_COHORTS = {
    "all_patient": {"manifest": "prognosis_all_patient.csv", "conditions": {}},
    "PE_positive": {
        "manifest": "prognosis_pe_positive.csv",
        "conditions": {"pe_positive_nlp": "TRUE"},
    },
}
IDENTITY_COLUMNS = ("patient_id", "study_id", "split", "image_path")
TRACE_COLUMNS = ("person_id", "image_id", "impression_id", "note_id", "procedure_datetime")


def _binary(value: Any) -> str:
    """Encode an INSPECT label as 1/0, leaving censored and missing values empty.

    An empty cell is masked out of the loss by the dataset; it is never a zero.
    """
    state = normalize_binary(value)
    return "1" if state == "TRUE" else "0" if state == "FALSE" else ""


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    temporary.replace(path)
    return path


def _base_row(record: StudyRecord, image_path: str | None = None) -> dict[str, Any]:
    return {
        "patient_id": record.patient_id,
        "study_id": record.study_id,
        "split": record.split,
        "image_path": image_path if image_path is not None else record.image_path,
        "person_id": record.patient_id,
        "image_id": record.study_id,
        "impression_id": record.impression_id,
        "note_id": record.note_id,
        "procedure_datetime": record.procedure_datetime,
    }


def _diagnosis_sources(
    label_map: Mapping[str, str], aliases: Mapping[str, str]
) -> dict[str, str]:
    """Return ``manifest target -> raw INSPECT label`` without silently remapping one."""
    result = {str(target): str(source) for source, target in label_map.items()}
    for target, source in aliases.items():
        target_text, source_text = str(target), str(source)
        previous = result.get(target_text)
        if previous is not None and previous != source_text:
            raise ValueError(
                f"diagnosis target {target_text!r} maps to both {previous!r} and {source_text!r}"
            )
        result[target_text] = source_text
    if not result:
        raise ValueError("at least one diagnosis label mapping is required")
    return result


def _outcome_columns(outcomes: Sequence[str]) -> list[str]:
    fields: list[str] = []
    for outcome in outcomes:
        fields.extend(
            (
                outcome,
                f"{outcome}_status",
                f"{outcome}_observed",
                f"{outcome}_censored",
                f"{outcome}_time_to_event",
                f"{outcome}_source_censor_flag",
            )
        )
    return fields


def _outcome_time_columns(outcome: str) -> tuple[str, str]:
    lower = outcome.lower()
    if "mortality" in lower:
        return "tte_mortality", "is_censored_mortality"
    if "readmission" in lower:
        return "tte_readmission", "is_censored_readmission"
    if lower.endswith("_ph") or "_ph_" in lower:
        return "tte_PH", "is_censored_PH"
    return "", ""


def _outcome_values(record: StudyRecord, outcomes: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for outcome in outcomes:
        observed = mortality_outcome(record.labels.get(outcome))
        time_column, censor_column = _outcome_time_columns(outcome)
        values.update(
            {
                outcome: observed["event"],
                f"{outcome}_status": observed["status"],
                f"{outcome}_observed": observed["observed"],
                f"{outcome}_censored": observed["censored"],
                f"{outcome}_time_to_event": record.labels.get(time_column, "") if time_column else "",
                f"{outcome}_source_censor_flag": record.labels.get(censor_column, "") if censor_column else "",
            }
        )
    return values


def _cohort_specs(value: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    raw = DEFAULT_PROGNOSIS_COHORTS if value is None else value
    result: dict[str, dict[str, Any]] = {}
    for name, specification in raw.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"prognosis cohort {name!r} must be a mapping")
        filename = str(specification.get("manifest") or "").strip()
        path = Path(filename)
        if not filename or path.name != filename or path.suffix.lower() != ".csv":
            raise ValueError(f"prognosis cohort {name!r} needs a simple .csv manifest name")
        conditions = dict(specification.get("conditions") or {})
        result[str(name)] = {"manifest": filename, "conditions": conditions}
    if not result:
        raise ValueError("at least one prognosis cohort is required")
    return result


def _prognosis_rows(
    index_studies: Mapping[str, StudyRecord],
    *,
    outcomes: Sequence[str],
    diagnosis_sources: Mapping[str, str],
    image_paths: Mapping[str, str],
    clinical_features: Mapping[str, Mapping[str, Any]],
    clinical_columns: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patient_id in sorted(index_studies):
        record = index_studies[patient_id]
        base = _base_row(record, image_paths.get(record.study_id, record.image_path))
        rows.append(
            {
                **base,
                "index_datetime": record.procedure_datetime,
                **_outcome_values(record, outcomes),
                "has_ehr_crosswalk": "TRUE" if record.has_ehr_crosswalk else "FALSE",
                **{
                    target: _binary(record.labels.get(source))
                    for target, source in diagnosis_sources.items()
                },
                **{
                    column: clinical_features.get(record.study_id, {}).get(column, "")
                    for column in clinical_columns
                },
            }
        )
    if len({row["patient_id"] for row in rows}) != len(rows):
        raise ValueError("prognosis cohort contains duplicate patients")
    return rows


def build_manifests(
    records: Sequence[StudyRecord],
    output_dir: str | Path,
    *,
    label_map: Mapping[str, str] | None = None,
    diagnosis_label_aliases: Mapping[str, str] | None = None,
    prognosis_label: str = DEFAULT_PROGNOSIS_LABEL,
    prognosis_conditions: Mapping[str, str] | None = None,
    prognosis_outcomes: Sequence[str] | None = None,
    prognosis_cohorts: Mapping[str, Mapping[str, Any]] | None = None,
    image_paths: Mapping[str, str] | None = None,
    include_report_text: bool = True,
    clinical_features: Mapping[str, Mapping[str, Any]] | None = None,
    clinical_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Write every derived manifest for one cohort. Returns a per-manifest summary.

    ``image_paths`` overrides the raw NIfTI path per study id, so a preprocessed cache
    can be pointed at without rebuilding the cohort. When it is absent the manifests
    reference the raw read-only volumes.
    """
    directory = Path(output_dir)
    mapping = dict(label_map or DEFAULT_LABEL_MAP)
    aliases = dict(
        DEFAULT_DIAGNOSIS_LABEL_ALIASES
        if diagnosis_label_aliases is None
        else diagnosis_label_aliases
    )
    diagnosis_sources = _diagnosis_sources(mapping, aliases)
    conditions = dict(DEFAULT_PROGNOSIS_CONDITIONS if prognosis_conditions is None else prognosis_conditions)
    outcomes = tuple(dict.fromkeys(str(value) for value in (prognosis_outcomes or DEFAULT_PROGNOSIS_OUTCOMES)))
    if not outcomes:
        raise ValueError("at least one prognosis outcome is required")
    cohorts = _cohort_specs(prognosis_cohorts)
    overrides = dict(image_paths or {})
    feature_values = dict(clinical_features or {})
    feature_columns = tuple(str(column) for column in clinical_columns)
    reserved = set(IDENTITY_COLUMNS) | set(TRACE_COLUMNS) | {"provenance", "has_ehr_crosswalk"}
    collisions = sorted(reserved & set(feature_columns))
    if collisions:
        raise ValueError("clinical columns collide with manifest columns: " + ", ".join(collisions))
    summary: dict[str, Any] = {}

    label_columns = list(diagnosis_sources)
    ctpa_fields = [*IDENTITY_COLUMNS, *TRACE_COLUMNS[3:], "provenance", "has_ehr_crosswalk"]
    ctpa_rows = []
    diagnosis_rows = []
    paired_rows = []
    report_rows = []
    for record in records:
        image_path = overrides.get(record.study_id, record.image_path)
        base = _base_row(record, image_path)
        clinical = {
            column: feature_values.get(record.study_id, {}).get(column, "")
            for column in feature_columns
        }
        ctpa_rows.append({**base, "provenance": record.provenance,
                          "has_ehr_crosswalk": "TRUE" if record.has_ehr_crosswalk else "FALSE", **clinical})
        labels = {
            target: _binary(record.labels.get(source))
            for target, source in diagnosis_sources.items()
        }
        diagnosis_rows.append({**base, **labels,
                               "has_ehr_crosswalk": "TRUE" if record.has_ehr_crosswalk else "FALSE", **clinical})
        paired = {**base, "report_id": record.impression_id}
        if include_report_text:
            paired["report_text"] = record.report_text
        paired_rows.append(paired)
        report_rows.append(
            {
                "patient_id": record.patient_id,
                "study_id": record.study_id,
                "report_id": record.impression_id,
                "split": record.split,
                "report_text": record.report_text,
            }
        )

    summary["ctpa"] = _write(directory / "ctpa.csv", ctpa_rows, [*ctpa_fields, *feature_columns])
    summary["diagnosis"] = _write(
        directory / "diagnosis.csv",
        diagnosis_rows,
        [*IDENTITY_COLUMNS, *TRACE_COLUMNS[2:], *label_columns, "has_ehr_crosswalk", *feature_columns],
    )
    paired_fields = [*IDENTITY_COLUMNS, *TRACE_COLUMNS[2:], "report_id"]
    if include_report_text:
        paired_fields.append("report_text")
    summary["paired_reports"] = _write(directory / "paired_reports.csv", paired_rows, paired_fields)
    summary["reports"] = _write(
        directory / "reports.csv",
        report_rows,
        ["patient_id", "study_id", "report_id", "split", "report_text"],
    )

    legacy_index_studies = first_index_study(records, conditions=conditions)
    prognosis_rows = _prognosis_rows(
        legacy_index_studies,
        outcomes=outcomes,
        diagnosis_sources=diagnosis_sources,
        image_paths=overrides,
        clinical_features=feature_values,
        clinical_columns=feature_columns,
    )
    # Keep the historical 30-day aliases so existing prognosis configs still run on the
    # legacy manifest.  The seven canonical outcome columns above are the new contract.
    for row in prognosis_rows:
        record = legacy_index_studies[row["patient_id"]]
        outcome = mortality_outcome(record.labels.get(prognosis_label))
        row.update(
            {
                "mortality_30d_status": outcome["status"],
                "mortality_30d": outcome["event"],
                "mortality_30d_observed": outcome["observed"],
                "mortality_30d_censored": outcome["censored"],
                "tte_mortality": record.labels.get("tte_mortality", ""),
                "is_censored_mortality": record.labels.get("is_censored_mortality", ""),
            }
        )
    prognosis_fields = [
        *IDENTITY_COLUMNS,
        *TRACE_COLUMNS[2:],
        "index_datetime",
        *label_columns,
        *_outcome_columns(outcomes),
        "mortality_30d_status",
        "mortality_30d",
        "mortality_30d_observed",
        "mortality_30d_censored",
        "tte_mortality",
        "is_censored_mortality",
        "has_ehr_crosswalk",
        *feature_columns,
    ]
    summary["prognosis"] = _write(
        directory / "prognosis.csv",
        prognosis_rows,
        prognosis_fields,
    )
    summary["prognosis_cohorts"] = {}
    cohort_fields = [
        *IDENTITY_COLUMNS,
        *TRACE_COLUMNS[2:],
        "index_datetime",
        *label_columns,
        *_outcome_columns(outcomes),
        "has_ehr_crosswalk",
        *feature_columns,
    ]
    for cohort_name, specification in cohorts.items():
        selected = first_index_study(records, conditions=specification["conditions"])
        rows = _prognosis_rows(
            selected,
            outcomes=outcomes,
            diagnosis_sources=diagnosis_sources,
            image_paths=overrides,
            clinical_features=feature_values,
            clinical_columns=feature_columns,
        )
        summary["prognosis_cohorts"][cohort_name] = _write(
            directory / specification["manifest"], rows, cohort_fields
        )
    summary["cohort_definitions"] = {
        "ctpa": "every eligible study after filtering and QC",
        "diagnosis": "every eligible study, native PE labels",
        "prognosis": (
            "earliest study per patient satisfying "
            + ", ".join(f"{key}={value}" for key, value in sorted(conditions.items()))
        ),
        "prognosis_cohorts": {
            name: {
                "manifest": specification["manifest"],
                "conditions": dict(specification["conditions"]),
            }
            for name, specification in cohorts.items()
        },
        "paired_reports": "every eligible study paired with its impression",
        "label_map": mapping,
        "diagnosis_label_aliases": aliases,
        "prognosis_label": prognosis_label,
        "prognosis_outcomes": list(outcomes),
        "clinical_columns": list(feature_columns),
    }
    return summary


def _write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    write_csv(path, rows, fields)
    return {
        "path": str(path),
        "rows": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "columns": list(fields),
    }
