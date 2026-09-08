"""Derived manifest construction for the project's training stages.

Ported from /mnt/pe_study `build_inspect_manifest.py`, `build_inspect_500_phase1_audit.py`
and `build_prognosis_cohort.py`, unified so one cohort pass emits every manifest the
pipeline consumes, all from the same eligible study set.

Emitted under ``<derived>/datasets/<profile>/manifests/``:

    ctpa.csv            every eligible study; the CTPA-level manifest (segmentation, DAPT)
    diagnosis.csv       the diagnosis cohort with native PE labels
    prognosis.csv       one index study per patient (confirmed acute PE), mortality fields
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
DEFAULT_PROGNOSIS_LABEL = "1_month_mortality"
DEFAULT_PROGNOSIS_CONDITIONS = {"pe_positive_nlp": "TRUE", "pe_acute": "TRUE"}
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


def build_manifests(
    records: Sequence[StudyRecord],
    output_dir: str | Path,
    *,
    label_map: Mapping[str, str] | None = None,
    prognosis_label: str = DEFAULT_PROGNOSIS_LABEL,
    prognosis_conditions: Mapping[str, str] | None = None,
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
    conditions = dict(DEFAULT_PROGNOSIS_CONDITIONS if prognosis_conditions is None else prognosis_conditions)
    overrides = dict(image_paths or {})
    feature_values = dict(clinical_features or {})
    feature_columns = tuple(str(column) for column in clinical_columns)
    reserved = set(IDENTITY_COLUMNS) | set(TRACE_COLUMNS) | {"provenance", "has_ehr_crosswalk"}
    collisions = sorted(reserved & set(feature_columns))
    if collisions:
        raise ValueError("clinical columns collide with manifest columns: " + ", ".join(collisions))
    summary: dict[str, Any] = {}

    label_columns = list(mapping.values())
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
        labels = {target: _binary(record.labels.get(source)) for source, target in mapping.items()}
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

    index_studies = first_index_study(records, conditions=conditions)
    prognosis_rows = []
    for patient_id in sorted(index_studies):
        record = index_studies[patient_id]
        outcome = mortality_outcome(record.labels.get(prognosis_label))
        base = _base_row(record, overrides.get(record.study_id, record.image_path))
        prognosis_rows.append(
            {
                **base,
                "index_datetime": record.procedure_datetime,
                "mortality_30d_status": outcome["status"],
                "mortality_30d": outcome["event"],
                "mortality_30d_observed": outcome["observed"],
                "mortality_30d_censored": outcome["censored"],
                "tte_mortality": record.labels.get("tte_mortality", ""),
                "is_censored_mortality": record.labels.get("is_censored_mortality", ""),
                "has_ehr_crosswalk": "TRUE" if record.has_ehr_crosswalk else "FALSE",
                **{target: _binary(record.labels.get(source)) for source, target in mapping.items()},
                **{column: feature_values.get(record.study_id, {}).get(column, "") for column in feature_columns},
            }
        )
    if len({row["patient_id"] for row in prognosis_rows}) != len(prognosis_rows):
        raise ValueError("prognosis cohort contains duplicate patients")
    summary["prognosis"] = _write(
        directory / "prognosis.csv",
        prognosis_rows,
        [
            *IDENTITY_COLUMNS,
            *TRACE_COLUMNS[2:],
            "index_datetime",
            *label_columns,
            "mortality_30d_status",
            "mortality_30d",
            "mortality_30d_observed",
            "mortality_30d_censored",
            "tte_mortality",
            "is_censored_mortality",
            "has_ehr_crosswalk",
            *feature_columns,
        ],
    )
    summary["cohort_definitions"] = {
        "ctpa": "every eligible study after filtering and QC",
        "diagnosis": "every eligible study, native PE labels",
        "prognosis": (
            "earliest study per patient satisfying "
            + ", ".join(f"{key}={value}" for key, value in sorted(conditions.items()))
        ),
        "paired_reports": "every eligible study paired with its impression",
        "label_map": mapping,
        "prognosis_label": prognosis_label,
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
