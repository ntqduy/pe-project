from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def silver_validation_metrics(
    rows: Sequence[Mapping[str, Any]],
    target_classes: Mapping[str, int],
    *,
    binary_threshold: float = 0.5,
    minimum_samples: int = 2,
) -> dict[str, dict[str, Any]]:
    """Calculate target metrics from globally gathered accepted validation predictions."""

    try:
        from sklearn import metrics
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for silver validation metrics") from exc
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        target = str(row.get("target") or "")
        if target in target_classes:
            grouped[target].append(row)
    output: dict[str, dict[str, Any]] = {}
    for target, classes in target_classes.items():
        selected = grouped.get(target, [])
        if len(selected) < minimum_samples:
            output[target] = {
                "status": "unavailable",
                "reason": f"insufficient_samples:{len(selected)}<{minimum_samples}",
                "samples": len(selected),
            }
            continue
        truth = np.asarray([int(row["y_true"]) for row in selected], dtype=int)
        observed = np.unique(truth)
        if observed.size < 2:
            output[target] = {
                "status": "unavailable",
                "reason": "single_observed_class",
                "samples": len(selected),
                "observed_classes": observed.tolist(),
            }
            continue
        if int(classes) == 1:
            probability = np.asarray([float(row["scores"][0]) for row in selected], dtype=float)
            prediction = (probability >= binary_threshold).astype(int)
            output[target] = {
                "status": "available",
                "samples": len(selected),
                "auroc": float(metrics.roc_auc_score(truth, probability)),
                "auprc": float(metrics.average_precision_score(truth, probability)),
                "f1": float(metrics.f1_score(truth, prediction, zero_division=0)),
                "threshold": float(binary_threshold),
            }
        else:
            scores = np.asarray([row["scores"] for row in selected], dtype=float)
            if scores.shape != (len(selected), int(classes)):
                raise ValueError(f"invalid multiclass score shape for {target}: {scores.shape}")
            prediction = scores.argmax(axis=1)
            output[target] = {
                "status": "available",
                "samples": len(selected),
                "macro_f1": float(metrics.f1_score(truth, prediction, average="macro", zero_division=0)),
                "balanced_accuracy": float(metrics.balanced_accuracy_score(truth, prediction)),
                "observed_classes": observed.tolist(),
            }
    return output
