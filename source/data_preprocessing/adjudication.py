"""Patient-level label adjudication for INSPECT study labels.

Ported from /mnt/pe_study `create_inspect_subset.aggregate_patient_value`,
`build_inspect_500_phase1_audit.aggregate_patient_label` and
`build_prognosis_cohort.mortality_fields`, which independently reimplemented the same
precedence. One implementation lives here.

Precedence for a binary INSPECT label across a patient's studies:

    TRUE  >  CENSORED  >  FALSE  >  MISSING

TRUE wins because INSPECT labels are report-derived positives: one positive study
makes the patient positive. CENSORED outranks FALSE because a censored outcome is not
an observed negative, and collapsing it into FALSE would invent survivors.

Adjudication here is label reconciliation, *not* silver-label adjudication between
LLM providers -- that is `source/silver/adjudicator.py` and is a different question.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .sources import StudyRecord

TRUE_TOKENS = {"1", "TRUE", "T", "YES", "Y"}
FALSE_TOKENS = {"0", "FALSE", "F", "NO", "N"}
CENSORED_TOKENS = {"CENSORED"}
PRECEDENCE = ("TRUE", "CENSORED", "FALSE", "MISSING")


def normalize_binary(value: Any) -> str:
    """Map any INSPECT label spelling onto TRUE / FALSE / CENSORED / MISSING."""
    text = str(value or "").strip()
    if not text:
        return "MISSING"
    upper = text.upper()
    if upper in TRUE_TOKENS:
        return "TRUE"
    if upper in FALSE_TOKENS:
        return "FALSE"
    if upper in CENSORED_TOKENS:
        return "CENSORED"
    if upper == "MISSING":
        return "MISSING"
    return text


def is_true(value: Any) -> bool:
    return normalize_binary(value) == "TRUE"


def adjudicate_binary(values: Sequence[Any]) -> str:
    """Reduce one label across a patient's studies using the documented precedence."""
    normalized = {normalize_binary(value) for value in values}
    for state in PRECEDENCE:
        if state in normalized:
            return state
    return "MISSING"


@dataclass(frozen=True)
class PatientLabel:
    patient_id: str
    split: str
    studies: int
    values: dict[str, str]

    def stratum(self, columns: Sequence[str]) -> tuple[str, ...]:
        return tuple(self.values.get(column, "MISSING") for column in columns)


def patient_labels(
    records: Sequence[StudyRecord], columns: Sequence[str]
) -> list[PatientLabel]:
    """Adjudicate ``columns`` per patient. Raises if a patient crosses official splits."""
    grouped: dict[str, list[StudyRecord]] = {}
    for record in records:
        grouped.setdefault(record.patient_id, []).append(record)
    labels: list[PatientLabel] = []
    for patient_id in sorted(grouped):
        patient_records = grouped[patient_id]
        splits = {record.split for record in patient_records}
        if len(splits) > 1:
            raise ValueError(
                f"patient {patient_id} appears in multiple official splits: {sorted(splits)}"
            )
        labels.append(
            PatientLabel(
                patient_id=patient_id,
                split=patient_records[0].split,
                studies=len(patient_records),
                values={
                    column: adjudicate_binary([record.labels.get(column) for record in patient_records])
                    for column in columns
                },
            )
        )
    return labels


def mortality_outcome(value: Any) -> dict[str, str]:
    """Split an INSPECT mortality label into event / observed / censored fields.

    A censored or missing outcome is never encoded as a survivor: ``event`` stays empty
    and ``observed`` stays 0, so a binary analysis has to drop the row explicitly rather
    than silently counting it as alive.
    """
    state = normalize_binary(value)
    if state == "TRUE":
        return {"status": "TRUE", "event": "1", "observed": "1", "censored": "0"}
    if state == "FALSE":
        return {"status": "FALSE", "event": "0", "observed": "1", "censored": "0"}
    if state == "CENSORED":
        return {"status": "CENSORED", "event": "", "observed": "0", "censored": "1"}
    return {"status": "MISSING", "event": "", "observed": "0", "censored": "0"}


def first_index_study(
    records: Sequence[StudyRecord],
    *,
    conditions: Mapping[str, str] | None = None,
) -> dict[str, StudyRecord]:
    """Select each patient's earliest study satisfying ``conditions`` (label == value).

    This is the prognosis index-event rule from
    /mnt/pe_study `build_prognosis_cohort.select_first_acute_pe`, generalized so the
    cohort definition lives in the dataset profile instead of in the code.
    """
    required = {key: normalize_binary(value) for key, value in (conditions or {}).items()}
    grouped: dict[str, list[StudyRecord]] = {}
    for record in records:
        if any(normalize_binary(record.labels.get(key)) != value for key, value in required.items()):
            continue
        grouped.setdefault(record.patient_id, []).append(record)
    return {
        patient_id: sorted(
            patient_records,
            key=lambda record: (record.procedure_datetime or "9999", record.impression_id),
        )[0]
        for patient_id, patient_records in grouped.items()
    }
