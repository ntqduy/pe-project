"""Cohort size, split composition and patient-level leakage checks."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .loaders import DatasetBundle


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("patient_id") or "").strip(),
        str(row.get("study_id") or "").strip(),
        str(row.get("split") or "").strip().lower(),
    )


def cohort_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Patients, studies and studies-per-patient for one manifest."""
    patients: dict[str, set[str]] = defaultdict(set)
    split_patients: dict[str, set[str]] = defaultdict(set)
    split_studies: Counter[str] = Counter()
    for row in rows:
        patient, study, split = _identity(row)
        if not patient or not study:
            continue
        patients[patient].add(study)
        if split:
            split_patients[split].add(patient)
            split_studies[split] += 1
    per_patient = sorted(len(studies) for studies in patients.values())
    return {
        "patients": len(patients),
        "studies": sum(per_patient),
        "studies_per_patient": {
            "min": per_patient[0] if per_patient else 0,
            "median": per_patient[len(per_patient) // 2] if per_patient else 0,
            "max": per_patient[-1] if per_patient else 0,
            "multi_study_patients": sum(1 for count in per_patient if count > 1),
        },
        "splits": {
            split: {"patients": len(split_patients[split]), "studies": split_studies[split]}
            for split in sorted(split_studies)
        },
    }


def leakage_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A patient in more than one split is the single most damaging dataset defect.

    Reported as an explicit list, not a boolean, so the offending patients can be traced.
    """
    splits_by_patient: dict[str, set[str]] = defaultdict(set)
    studies: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    for row in rows:
        patient, study, split = _identity(row)
        if not patient or not study:
            continue
        if split:
            splits_by_patient[patient].add(split)
        studies[study] += 1
        pairs[(patient, study)] += 1
    crossing = sorted(
        patient for patient, values in splits_by_patient.items() if len(values) > 1
    )
    return {
        "patients_crossing_splits": len(crossing),
        "patients_crossing_splits_examples": crossing[:20],
        "duplicate_study_ids": sorted(key for key, count in studies.items() if count > 1)[:20],
        "duplicate_study_id_count": sum(1 for count in studies.values() if count > 1),
        "duplicate_patient_study_pairs": sum(1 for count in pairs.values() if count > 1),
        "ok": not crossing and all(count == 1 for count in pairs.values()),
    }


def cohort_membership_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Inclusion/exclusion accounting from prognosis_cohort_membership.csv."""
    by_cohort: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cohort = str(row.get("cohort_name") or "unknown")
        status = str(row.get("inclusion_status") or "unknown")
        by_cohort[cohort][status] += 1
        if status != "included":
            reasons[cohort][str(row.get("exclusion_reason") or "unspecified")] += 1
    return {
        cohort: {
            "counts": dict(sorted(counts.items())),
            "exclusion_reasons": dict(sorted(reasons[cohort].items(), key=lambda item: -item[1])),
        }
        for cohort, counts in sorted(by_cohort.items())
    }


def analyse(bundle: DatasetBundle) -> dict[str, Any]:
    output: dict[str, Any] = {"per_manifest": {}, "leakage": {}}
    for name in ("ctpa.csv", "diagnosis.csv", "prognosis_all_patient.csv",
                 "prognosis_pe_positive.csv", "prognosis.csv", "paired_reports.csv"):
        rows = bundle.rows(name)
        if not rows:
            continue
        output["per_manifest"][name] = cohort_summary(rows)
        output["leakage"][name] = leakage_check(rows)
    if bundle.has("prognosis_cohort_membership.csv"):
        output["cohort_membership"] = cohort_membership_summary(
            bundle.rows("prognosis_cohort_membership.csv")
        )
    return output
