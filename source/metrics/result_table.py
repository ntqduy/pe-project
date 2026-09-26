"""Human-readable per-split result tables shared by task evaluation and zero-shot baselines.

``result.csv`` has one row per (target, split) with the same columns for every diagnosis or
prognosis run, so tables from different runs can be concatenated and compared directly:

    experiment, target, split, n_patients, n_studies, n_pos, n_neg,
    auroc, auroc_ci_low, auroc_ci_high, auprc, auprc_ci_low, auprc_ci_high,
    threshold, threshold_rule, sensitivity, specificity, ppv, npv, f1,
    balanced_accuracy, accuracy, brier, [calibration_intercept, calibration_slope], note

Cells that cannot be computed are left empty (never "nan"); ``note`` says why. Confidence
intervals are patient-bootstrap intervals and exist on the test split only. The full
metric payload, including intervals for every metric, stays in ``result.json``.
"""
from __future__ import annotations

import csv
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from .calibration import DEFAULT_CALIBRATION_MIN_EVENTS, calibration_unavailable_reason

SPLIT_ORDER = ("train", "validation", "test")
BASE_COLUMNS = (
    "experiment",
    "target",
    "split",
    "n_patients",
    "n_studies",
    "n_pos",
    "n_neg",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "auprc",
    "auprc_ci_low",
    "auprc_ci_high",
    "threshold",
    "threshold_rule",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
    "accuracy",
    "brier",
)
CALIBRATION_COLUMNS = ("calibration_intercept", "calibration_slope")
PREDICTION_COLUMNS = ("split", "target", "patient_id", "study_id", "y_true", "y_prob", "y_pred")
# Metrics shown with a bootstrap interval in the CSV; all intervals remain in result.json.
CI_METRICS = ("auroc", "auprc")
POINT_METRICS = (
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
    "accuracy",
    "brier",
)
# A split with fewer cases than this in either class is flagged as unstable.
DEFAULT_MIN_CLASS_COUNT = 5
# result.json keeps its historical threshold_source values (external evaluation checks
# threshold_source == "validation"); the CSV spells out the rule instead.
THRESHOLD_SOURCE_VALIDATION = "validation"
THRESHOLD_SOURCE_FALLBACK = "fallback_0.5_validation_single_class"
THRESHOLD_SOURCE_LOCKED = "internal_validation_artifact"
_CI_REASONS = {
    "no_evaluable_test_rows": "no evaluable test rows",
    "fewer_than_two_patients": "fewer than two patients",
}


def threshold_rule(threshold_source: str | None, method: str = "youden") -> str:
    """Readable name of how the decision threshold was chosen."""
    source = str(threshold_source or "")
    if source == THRESHOLD_SOURCE_VALIDATION:
        return f"{method}_on_validation"
    if source == THRESHOLD_SOURCE_FALLBACK:
        return "default_0.5_validation_one_class"
    if source == THRESHOLD_SOURCE_LOCKED:
        return "locked_internal_validation"
    return source


def select_threshold(rows: Sequence[Mapping[str, Any]], method: str = "youden") -> tuple[float, str]:
    """Validation threshold and its source; 0.5 when validation holds a single class."""
    from .classification import select_threshold_on_validation

    truth = [int(row["y_true"]) for row in rows]
    if len(set(truth)) == 2:
        threshold = select_threshold_on_validation(
            truth, [float(row["y_prob"]) for row in rows], method=method
        )
        return float(threshold), THRESHOLD_SOURCE_VALIDATION
    return 0.5, THRESHOLD_SOURCE_FALLBACK


def split_summary(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """Case counts and prediction range of one split; the context every metric needs."""
    truth = [int(row["y_true"]) for row in rows]
    probability = [float(row["y_prob"]) for row in rows]
    predicted = [int(value >= float(threshold)) for value in probability]
    return {
        "n_patients": len({str(row["patient_id"]) for row in rows}),
        "n_studies": len(rows),
        "n_pos": sum(truth),
        "n_neg": len(truth) - sum(truth),
        "predicted_pos": sum(predicted),
        "predicted_neg": len(predicted) - sum(predicted),
        "prob_min": min(probability) if probability else None,
        "prob_max": max(probability) if probability else None,
    }


def metric_bundle(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
    threshold: float,
    *,
    with_ci: bool,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    calibration_min_events: int = DEFAULT_CALIBRATION_MIN_EVENTS,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Point metrics, plus a patient-bootstrap interval when ``with_ci``.

    Returns ``(metrics, reason)``; ``reason`` explains a missing interval for the whole
    split, per-metric gaps are in ``metrics[name]["ci_note"]``.
    """
    from .bootstrap import bootstrap_binary_predictions, bootstrap_prognosis_predictions
    from .classification import binary_classification_metrics
    from .prognosis import prognosis_metrics

    if not rows:
        return {}, "no_evaluable_test_rows"
    if stage == "prognosis":
        point_fn = partial(prognosis_metrics, calibration_min_events=calibration_min_events)
        bootstrap = partial(bootstrap_prognosis_predictions, calibration_min_events=calibration_min_events)
    else:
        point_fn = binary_classification_metrics
        bootstrap = bootstrap_binary_predictions

    def point_only() -> dict[str, dict[str, Any]]:
        point = point_fn(
            [int(row["y_true"]) for row in rows],
            [float(row["y_prob"]) for row in rows],
            threshold,
        )
        return {
            name: {"value": value, "ci_low": float("nan"), "ci_high": float("nan"), "valid_replicates": 0}
            for name, value in point.items()
            if name != "threshold"
        }

    if not with_ci:
        return point_only(), None
    if len({str(row["patient_id"]) for row in rows}) < 2:
        return point_only(), "fewer_than_two_patients"
    try:
        return bootstrap(rows, threshold, n_bootstrap=samples, confidence=confidence, seed=seed), None
    except (RuntimeError, ValueError) as exc:
        # Keep the point estimate when a small/single-class split cannot be resampled.
        return point_only(), f"bootstrap_failed:{type(exc).__name__}:{exc}"


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cell(value: Any, digits: int = 4) -> Any:
    number = _finite(value)
    return "" if number is None else round(number, digits)


def _metric_part(metrics: Mapping[str, Any], name: str, part: str = "value") -> Any:
    item = metrics.get(name)
    if isinstance(item, Mapping):
        return item.get(part)
    return item if part == "value" else None


def row_notes(
    split: str,
    metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    threshold: float,
    ci_reason: str | None = None,
    calibration_reason: str | None = None,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
) -> list[str]:
    """Plain explanations for every empty cell and every stability warning of one row."""
    notes: list[str] = []
    positives, negatives = int(summary.get("n_pos") or 0), int(summary.get("n_neg") or 0)
    if not positives + negatives:
        return ["no evaluable cases in this split"]
    if not positives or not negatives:
        undefined = [
            name
            for name in ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity")
            if _finite(_metric_part(metrics, name)) is None
        ]
        notes.append(
            f"only one class (n_pos={positives}, n_neg={negatives})"
            + (f" -> {'/'.join(undefined)} undefined" if undefined else "")
        )
    elif min(positives, negatives) < int(min_class_count):
        notes.append(f"few cases (n_pos={positives}, n_neg={negatives}): unstable estimate")
    predicted_pos = int(summary.get("predicted_pos") or 0)
    predicted_neg = int(summary.get("predicted_neg") or 0)
    if positives and negatives and (not predicted_pos or not predicted_neg):
        low, high = _finite(summary.get("prob_min")), _finite(summary.get("prob_max"))
        span = f" [{low:.4f}, {high:.4f}]" if low is not None and high is not None else ""
        side = "negative" if not predicted_pos else "positive"
        notes.append(f"threshold {threshold:.4f} vs y_prob range{span} -> every case predicted {side}")
    if not predicted_pos and _finite(_metric_part(metrics, "ppv")) is None:
        notes.append("no predicted positives -> PPV undefined")
    if not predicted_neg and _finite(_metric_part(metrics, "npv")) is None:
        notes.append("no predicted negatives -> NPV undefined")
    if calibration_reason and any(
        _finite(_metric_part(metrics, name)) is None for name in CALIBRATION_COLUMNS if name in metrics
    ):
        notes.append(calibration_reason)
    if split == "test":
        if ci_reason:
            text = _CI_REASONS.get(ci_reason, ci_reason.replace("bootstrap_failed:", "bootstrap failed: "))
            notes.append(f"no CI: {text}")
        else:
            for name in CI_METRICS:
                if _finite(_metric_part(metrics, name)) is None:
                    continue
                note = _metric_part(metrics, name, "ci_note")
                if note:
                    notes.append(f"no CI for {name.upper()}: {note}")
    return notes


def result_rows(
    *,
    experiment: str,
    target: str,
    splits: Mapping[str, Mapping[str, Any]],
    threshold: float,
    threshold_source: str | None,
    stage: str,
    threshold_method: str = "youden",
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
) -> list[dict[str, Any]]:
    """Rows for one target. ``splits[name]`` holds ``metrics``, ``summary``, ``ci_reason``
    and, for prognosis, ``calibration_reason``; absent splits are skipped."""
    rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        entry = splits.get(split)
        if entry is None:
            continue
        metrics = dict(entry.get("metrics") or {})
        summary = dict(entry.get("summary") or {})
        with_ci = split == "test"
        row: dict[str, Any] = {
            "experiment": experiment,
            "target": target,
            "split": split,
            "n_patients": summary.get("n_patients", ""),
            "n_studies": summary.get("n_studies", ""),
            "n_pos": summary.get("n_pos", ""),
            "n_neg": summary.get("n_neg", ""),
            "threshold": _cell(threshold),
            "threshold_rule": threshold_rule(threshold_source, threshold_method),
        }
        for name in CI_METRICS:
            row[name] = _cell(_metric_part(metrics, name))
            row[f"{name}_ci_low"] = _cell(_metric_part(metrics, name, "ci_low")) if with_ci else ""
            row[f"{name}_ci_high"] = _cell(_metric_part(metrics, name, "ci_high")) if with_ci else ""
        for name in POINT_METRICS:
            row[name] = _cell(_metric_part(metrics, name))
        if stage == "prognosis":
            for name in CALIBRATION_COLUMNS:
                row[name] = _cell(_metric_part(metrics, name))
        row["note"] = "; ".join(
            row_notes(
                split,
                metrics,
                summary,
                threshold=float(threshold),
                ci_reason=entry.get("ci_reason") if with_ci else None,
                calibration_reason=entry.get("calibration_reason") if stage == "prognosis" else None,
                min_class_count=min_class_count,
            )
        )
        rows.append(row)
    return rows


def calibration_reason(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_events: int = DEFAULT_CALIBRATION_MIN_EVENTS,
) -> str | None:
    if not rows:
        return None
    return calibration_unavailable_reason(
        [int(row["y_true"]) for row in rows],
        [float(row["y_prob"]) for row in rows],
        min_events=min_events,
    )


def result_columns(stage: str) -> tuple[str, ...]:
    return (*BASE_COLUMNS, *(CALIBRATION_COLUMNS if stage == "prognosis" else ()), "note")


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_result_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, stage: str) -> Path:
    return _write_csv(path, result_columns(stage), rows)


def prediction_rows(
    split: str,
    target: str,
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "target": target,
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "y_true": int(row["y_true"]),
            "y_prob": round(float(row["y_prob"]), 6),
            "y_pred": int(float(row["y_prob"]) >= float(threshold)),
            **{key: row[key] for key in row if key not in PREDICTION_COLUMNS and key != "target"},
        }
        for row in rows
    ]


def write_predictions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    extra = [key for row in rows for key in row if key not in PREDICTION_COLUMNS]
    columns = [*PREDICTION_COLUMNS, *dict.fromkeys(extra)]
    return _write_csv(path, columns, rows)


def read_prediction_file(path: Path, *, split: str | None = "test") -> list[dict[str, Any]]:
    """Read predictions.csv (or a legacy predictions.parquet), optionally one split only."""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"patient_id": str, "study_id": str})
    else:
        frame = pd.read_parquet(path)
    records = frame.to_dict(orient="records")
    if split is not None and "split" in frame.columns:
        records = [row for row in records if str(row.get("split")) == split]
    for row in records:
        row["patient_id"] = str(row["patient_id"])
        if "study_id" in row:
            row["study_id"] = str(row["study_id"])
    return records


def log_lines(
    target: str,
    split: str,
    summary: Mapping[str, Any],
    *,
    threshold: float,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
) -> list[str]:
    """One summary line per split plus WARNING lines for degenerate splits."""
    positives, negatives = int(summary.get("n_pos") or 0), int(summary.get("n_neg") or 0)
    if not positives + negatives:
        return [f"WARNING target={target} split={split}: no evaluable rows"]
    lines = [
        (
            f"target={target} split={split} patients={summary.get('n_patients')} "
            f"studies={summary.get('n_studies')} n_pos={positives} n_neg={negatives} "
            f"predicted_pos={summary.get('predicted_pos')} predicted_neg={summary.get('predicted_neg')}"
        )
    ]
    if not positives or not negatives:
        lines.append(
            f"WARNING target={target} split={split}: only one class "
            f"(n_pos={positives}, n_neg={negatives}); AUROC/AUPRC undefined"
        )
    elif min(positives, negatives) < int(min_class_count):
        lines.append(
            f"WARNING target={target} split={split}: fewer than {int(min_class_count)} cases in a class "
            f"(n_pos={positives}, n_neg={negatives}); metrics are unstable"
        )
    predicted_pos = int(summary.get("predicted_pos") or 0)
    predicted_neg = int(summary.get("predicted_neg") or 0)
    if not predicted_pos or not predicted_neg:
        side = "negative" if not predicted_pos else "positive"
        lines.append(
            f"WARNING target={target} split={split}: threshold {threshold:.4f} vs "
            f"y_prob [{float(summary.get('prob_min')):.4f}, {float(summary.get('prob_max')):.4f}]; "
            f"every case is predicted {side}"
        )
    return lines


__all__ = [
    "BASE_COLUMNS",
    "CALIBRATION_COLUMNS",
    "PREDICTION_COLUMNS",
    "calibration_reason",
    "log_lines",
    "metric_bundle",
    "prediction_rows",
    "read_prediction_file",
    "result_columns",
    "result_rows",
    "row_notes",
    "select_threshold",
    "split_summary",
    "threshold_rule",
    "write_predictions_csv",
    "write_result_csv",
]
