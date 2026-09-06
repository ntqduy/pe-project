from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _dependencies() -> tuple[Any, Any]:
    try:
        from sklearn import metrics
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for classification metrics") from exc
    return np, metrics


def select_threshold_on_validation(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    method: str = "youden",
) -> float:
    array, metrics = _dependencies()
    truth = array.asarray(y_true, dtype=int)
    probability = array.asarray(y_prob, dtype=float)
    if truth.shape != probability.shape or truth.size == 0 or len(array.unique(truth)) != 2:
        raise ValueError("validation threshold selection requires paired probabilities from both classes")
    if method == "youden":
        false_positive, true_positive, thresholds = metrics.roc_curve(truth, probability)
        finite = array.isfinite(thresholds)
        if not finite.any():
            raise ValueError("no finite validation threshold")
        score = true_positive[finite] - false_positive[finite]
        candidates = thresholds[finite]
        return float(candidates[int(array.argmax(score))])
    if method == "f1":
        precision, recall, thresholds = metrics.precision_recall_curve(truth, probability)
        if not len(thresholds):
            raise ValueError("no validation threshold candidates")
        f1 = 2 * precision[:-1] * recall[:-1] / array.maximum(precision[:-1] + recall[:-1], 1e-12)
        return float(thresholds[int(array.nanargmax(f1))])
    raise ValueError(f"unsupported threshold method: {method}")


def binary_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    array, metrics = _dependencies()
    truth = array.asarray(y_true, dtype=int)
    probability = array.asarray(y_prob, dtype=float)
    if truth.shape != probability.shape or truth.size == 0:
        raise ValueError("truth and probability must be non-empty paired arrays")
    if not array.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite and within [0,1]")
    prediction = (probability >= float(threshold)).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    divide = lambda numerator, denominator: float(numerator / denominator) if denominator else float("nan")
    both_classes = len(array.unique(truth)) == 2
    return {
        "auroc": float(metrics.roc_auc_score(truth, probability)) if both_classes else float("nan"),
        "auprc": float(metrics.average_precision_score(truth, probability)) if both_classes else float("nan"),
        "accuracy": float(metrics.accuracy_score(truth, prediction)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(truth, prediction)) if both_classes else float("nan"),
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
        "f1": float(metrics.f1_score(truth, prediction, zero_division=0)),
        "brier": float(metrics.brier_score_loss(truth, probability)),
        "threshold": float(threshold),
    }
