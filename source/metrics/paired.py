from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def align_predictions_by_patient(
    reference_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    patient_column: str = "patient_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]:
    """Restrict both prediction sets to their common patients, in a shared, stable order.

    Required for any counterfactual/ROI comparison (section 14): the two variants must be
    evaluated on the SAME held-out patients before a paired difference is meaningful.
    """
    def row_key(row: Mapping[str, Any]) -> tuple[str, str]:
        patient = str(row.get(patient_column, ""))
        study = str(row.get("study_id", ""))
        return patient, study

    reference_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in reference_rows:
        patient = str(row.get(patient_column, ""))
        if not patient:
            raise ValueError("paired alignment requires patient_id on every reference row")
        key = row_key(row)
        if key in reference_by_key:
            raise ValueError(f"duplicate reference patient/study prediction: {key}")
        reference_by_key[key] = dict(row)
    comparison_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in comparison_rows:
        patient = str(row.get(patient_column, ""))
        if not patient:
            raise ValueError("paired alignment requires patient_id on every comparison row")
        key = row_key(row)
        if key in comparison_by_key:
            raise ValueError(f"duplicate comparison patient/study prediction: {key}")
        comparison_by_key[key] = dict(row)
    common = sorted(set(reference_by_key) & set(comparison_by_key))
    common_patients = tuple(sorted({key[0] for key in common}))
    missing_from_comparison = sorted(set(reference_by_key) - set(comparison_by_key))
    missing_from_reference = sorted(set(comparison_by_key) - set(reference_by_key))
    if missing_from_comparison or missing_from_reference:
        raise ValueError(
            "paired comparison patient sets differ: "
            f"missing_from_comparison={missing_from_comparison} "
            f"missing_from_reference={missing_from_reference}"
        )
    if len(common_patients) < 2:
        raise ValueError("paired comparison requires at least two shared patients")
    target_mismatch = [
        key
        for key in common
        if "y_true" in reference_by_key[key]
        and "y_true" in comparison_by_key[key]
        and int(reference_by_key[key]["y_true"])
        != int(comparison_by_key[key]["y_true"])
    ]
    if target_mismatch:
        raise ValueError(
            "paired comparison target values differ for patients: "
            + ", ".join(f"{patient}/{study}" for patient, study in target_mismatch[:10])
        )
    return (
        [reference_by_key[key] for key in common],
        [comparison_by_key[key] for key in common],
        common_patients,
    )


def paired_patient_bootstrap(
    reference_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], Mapping[str, float]],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    patient_column: str = "patient_id",
) -> dict[str, dict[str, float | int]]:
    """Paired patient-level bootstrap for a counterfactual/ROI comparison against a reference.

    Resamples the SAME patients for both variants on each replicate, so the resulting delta
    distribution reflects within-patient pairing rather than two independent bootstraps. Reports
    each variant's point estimate plus the paired delta (comparison - reference) with its own CI.
    """
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("bootstrap count and confidence are invalid")
    reference, comparison, patients = align_predictions_by_patient(
        reference_rows, comparison_rows, patient_column=patient_column
    )
    reference_point = dict(metric_fn([dict(row) for row in reference]))
    comparison_point = dict(metric_fn([dict(row) for row in comparison]))
    names = sorted(set(reference_point) & set(comparison_point))
    if not names:
        raise ValueError("reference and comparison metric_fn outputs share no metric names")
    reference_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comparison_by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference:
        reference_by_patient[str(row[patient_column])].append(row)
    for row in comparison:
        comparison_by_patient[str(row[patient_column])].append(row)
    delta_samples: dict[str, list[float]] = {name: [] for name in names}
    generator = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        selected = generator.choice(patients, size=len(patients), replace=True)
        reference_replicate = [
            dict(row) for patient in selected for row in reference_by_patient[str(patient)]
        ]
        comparison_replicate = [
            dict(row) for patient in selected for row in comparison_by_patient[str(patient)]
        ]
        reference_result = metric_fn(reference_replicate)
        comparison_result = metric_fn(comparison_replicate)
        for name in names:
            reference_value = float(reference_result[name])
            comparison_value = float(comparison_result[name])
            if np.isfinite(reference_value) and np.isfinite(comparison_value):
                delta_samples[name].append(comparison_value - reference_value)
    alpha = (1 - confidence) / 2
    output: dict[str, dict[str, float | int]] = {}
    minimum_valid = max(20, n_bootstrap // 2)
    for name in names:
        valid = np.asarray(delta_samples[name], dtype=float)
        reference_value = float(reference_point[name])
        comparison_value = float(comparison_point[name])
        if not np.isfinite(reference_value) or not np.isfinite(comparison_value):
            output[name] = {
                "reference_value": reference_value,
                "comparison_value": comparison_value,
                "delta": float("nan"),
                "delta_ci_low": float("nan"),
                "delta_ci_high": float("nan"),
                "valid_replicates": int(valid.size),
                "sample_count": len(patients),
            }
            continue
        if valid.size < minimum_valid:
            raise RuntimeError(f"insufficient valid paired-bootstrap replicates for {name}: {valid.size}/{n_bootstrap}")
        output[name] = {
            "reference_value": reference_value,
            "comparison_value": comparison_value,
            "delta": comparison_value - reference_value,
            "delta_ci_low": float(np.quantile(valid, alpha)),
            "delta_ci_high": float(np.quantile(valid, 1 - alpha)),
            "valid_replicates": int(valid.size),
            "sample_count": len(patients),
        }
    return output
