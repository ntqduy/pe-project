from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any


class CrossValidationError(ValueError):
    pass


def stratified_patient_kfold_assignments(
    rows: Sequence[Mapping[str, Any]],
    k: int,
    seed: int,
    *,
    target_column: str,
    patient_column: str = "patient_id",
) -> dict[str, int]:
    """Deterministic patient-level stratification without ever splitting a patient."""

    labels: dict[str, str] = {}
    for row in rows:
        patient = str(row.get(patient_column, "")).strip()
        label = str(row.get(target_column, "")).strip()
        if not patient or not label:
            raise CrossValidationError("stratified folds require non-empty patient and target values")
        previous = labels.setdefault(patient, label)
        if previous != label:
            raise CrossValidationError(
                f"patient {patient} has inconsistent {target_column}: {previous!r} vs {label!r}"
            )
    if k < 2 or k > len(labels):
        raise CrossValidationError(f"invalid k={k} for {len(labels)} unique patients")
    by_class: dict[str, list[str]] = {}
    for patient, label in labels.items():
        by_class.setdefault(label, []).append(patient)
    too_small = {label: len(values) for label, values in by_class.items() if len(values) < k}
    if too_small:
        raise CrossValidationError(
            f"every class needs at least k patients for stratification: {too_small}"
        )
    assignments: dict[str, int] = {}
    for class_index, label in enumerate(sorted(by_class)):
        patients = sorted(by_class[label])
        random.Random(int(seed) * 1009 + class_index).shuffle(patients)
        for index, patient in enumerate(patients):
            assignments[patient] = index % k
    return assignments


def repeated_stratified_patient_splits(
    rows: Sequence[Mapping[str, Any]],
    k: int,
    repeats: int,
    seed: int,
    *,
    target_column: str,
    patient_column: str = "patient_id",
) -> list[dict[str, int]]:
    if repeats < 1:
        raise CrossValidationError("repeats must be at least 1")
    return [
        stratified_patient_kfold_assignments(
            rows,
            k,
            int(seed) * 1000 + repeat,
            target_column=target_column,
            patient_column=patient_column,
        )
        for repeat in range(repeats)
    ]


def patient_kfold_assignments(patients: Sequence[str], k: int, seed: int) -> dict[str, int]:
    """Deterministic patient-level K-fold assignment: every patient in exactly one fold."""
    ordered = sorted({str(patient) for patient in patients})
    if k < 2:
        raise CrossValidationError("k must be at least 2")
    if k > len(ordered):
        raise CrossValidationError(f"k={k} exceeds the number of unique patients ({len(ordered)})")
    shuffled = list(ordered)
    random.Random(int(seed)).shuffle(shuffled)
    return {patient: index % k for index, patient in enumerate(shuffled)}


def repeated_kfold_patient_splits(
    rows: Sequence[Mapping[str, Any]],
    k: int,
    repeats: int,
    seed: int,
    *,
    patient_column: str = "patient_id",
) -> list[dict[str, int]]:
    """Independent, deterministic K-fold assignments for repeated CV on a small prognosis cohort.

    Each element is one complete patient->fold assignment (section 13: repeated or nested CV
    must be supported for small event counts). Repeats use distinct, deterministic seeds derived
    from ``seed`` so the whole set is reproducible from one integer.
    """
    if repeats < 1:
        raise CrossValidationError("repeats must be at least 1")
    patients = [str(row[patient_column]) for row in rows]
    return [
        patient_kfold_assignments(patients, k, seed=int(seed) * 1000 + repeat)
        for repeat in range(int(repeats))
    ]


def nested_kfold_patient_splits(
    rows: Sequence[Mapping[str, Any]],
    outer_k: int,
    inner_k: int,
    seed: int,
    *,
    patient_column: str = "patient_id",
) -> list[dict[str, Any]]:
    """Nested CV: each outer fold holds out test patients; the rest are split into inner folds.

    Returns one entry per outer fold: ``{"outer_fold", "test_patients", "inner_folds"}``, where
    ``inner_folds`` maps each non-test patient to an inner fold index (used for train/validation
    within that outer fold). No patient appears in both a fold's test set and its inner folds.
    """
    patients = [str(row[patient_column]) for row in rows]
    outer = patient_kfold_assignments(patients, outer_k, seed)
    unique_patients = sorted(set(patients))
    result: list[dict[str, Any]] = []
    for outer_fold in range(outer_k):
        test_patients = tuple(patient for patient in unique_patients if outer[patient] == outer_fold)
        remaining = [patient for patient in unique_patients if outer[patient] != outer_fold]
        inner = patient_kfold_assignments(remaining, inner_k, seed=int(seed) * 1000 + outer_fold)
        result.append({"outer_fold": outer_fold, "test_patients": test_patients, "inner_folds": inner})
    return result


def assign_fold_column(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
    *,
    patient_column: str = "patient_id",
    fold_column: str = "fold",
) -> list[dict[str, Any]]:
    """Apply a patient->fold assignment to manifest rows, refusing silent gaps."""
    output: list[dict[str, Any]] = []
    for row in rows:
        patient = str(row[patient_column])
        if patient not in assignments:
            raise CrossValidationError(f"patient not present in fold assignment: {patient}")
        updated = dict(row)
        updated[fold_column] = assignments[patient]
        output.append(updated)
    return output
