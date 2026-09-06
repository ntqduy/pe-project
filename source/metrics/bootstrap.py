from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def patient_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], Mapping[str, float]],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    patient_column: str = "patient_id",
) -> dict[str, dict[str, float | int]]:
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("bootstrap count and confidence are invalid")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        patient = str(row.get(patient_column, ""))
        if not patient:
            raise ValueError("patient-level bootstrap requires patient_id on every row")
        grouped[patient].append(dict(row))
    patients = sorted(grouped)
    if len(patients) < 2:
        raise ValueError("patient-level bootstrap requires at least two patients")
    point = dict(metric_fn([dict(row) for row in rows]))
    samples: dict[str, list[float]] = {name: [] for name in point}
    generator = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        selected = generator.choice(patients, size=len(patients), replace=True)
        replicate: list[dict[str, Any]] = []
        for draw, patient in enumerate(selected):
            for row in grouped[str(patient)]:
                copied = dict(row)
                copied["_bootstrap_draw"] = draw
                replicate.append(copied)
        result = metric_fn(replicate)
        for name in samples:
            value = float(result[name])
            if np.isfinite(value):
                samples[name].append(value)
    alpha = (1 - confidence) / 2
    output: dict[str, dict[str, float | int]] = {}
    for name, value in point.items():
        valid = np.asarray(samples[name], dtype=float)
        if not np.isfinite(float(value)):
            output[name] = {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "valid_replicates": int(valid.size)}
        elif valid.size < max(20, n_bootstrap // 2):
            raise RuntimeError(f"insufficient valid bootstrap replicates for {name}: {valid.size}/{n_bootstrap}")
        else:
            output[name] = {
                "value": float(value),
                "ci_low": float(np.quantile(valid, alpha)),
                "ci_high": float(np.quantile(valid, 1 - alpha)),
                "valid_replicates": int(valid.size),
            }
    return output


def bootstrap_binary_predictions(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    from .classification import binary_classification_metrics

    def calculate(items: list[dict[str, Any]]) -> Mapping[str, float]:
        metrics = binary_classification_metrics(
            [int(item["y_true"]) for item in items],
            [float(item["y_prob"]) for item in items],
            threshold,
        )
        metrics.pop("threshold")
        return metrics

    return patient_bootstrap(rows, calculate, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)


def bootstrap_prognosis_predictions(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    from .prognosis import prognosis_metrics

    def calculate(items: list[dict[str, Any]]) -> Mapping[str, float]:
        metrics = prognosis_metrics(
            [int(item["y_true"]) for item in items],
            [float(item["y_prob"]) for item in items],
            threshold,
        )
        metrics.pop("threshold")
        return metrics

    return patient_bootstrap(rows, calculate, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
