from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def dice_score(prediction: Any, target: Any, epsilon: float = 1e-8) -> float:
    predicted = np.asarray(prediction).astype(bool)
    truth = np.asarray(target).astype(bool)
    if predicted.shape != truth.shape:
        raise ValueError("segmentation arrays must have equal shapes")
    denominator = predicted.sum() + truth.sum()
    return 1.0 if denominator == 0 else float((2 * np.logical_and(predicted, truth).sum() + epsilon) / (denominator + epsilon))


def _surface(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_erosion
    except ModuleNotFoundError as exc:
        raise RuntimeError("SciPy is required for surface metrics") from exc
    return np.logical_xor(mask, binary_erosion(mask, border_value=0))


def surface_metrics(
    prediction: Any,
    target: Any,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    tolerance_mm: float = 1.0,
) -> dict[str, float]:
    try:
        from scipy.ndimage import distance_transform_edt
    except ModuleNotFoundError as exc:
        raise RuntimeError("SciPy is required for surface metrics") from exc
    predicted = np.asarray(prediction).astype(bool)
    truth = np.asarray(target).astype(bool)
    if predicted.shape != truth.shape:
        raise ValueError("segmentation arrays must have equal shapes")
    if not predicted.any() and not truth.any():
        return {"nsd": 1.0, "hd95": 0.0}
    if not predicted.any() or not truth.any():
        return {"nsd": 0.0, "hd95": float("inf")}
    predicted_surface = _surface(predicted)
    truth_surface = _surface(truth)
    distance_to_truth = distance_transform_edt(~truth_surface, sampling=tuple(spacing))[predicted_surface]
    distance_to_prediction = distance_transform_edt(~predicted_surface, sampling=tuple(spacing))[truth_surface]
    distances = np.concatenate((distance_to_truth, distance_to_prediction))
    return {"nsd": float((distances <= tolerance_mm).mean()), "hd95": float(np.quantile(distances, 0.95))}


def segmentation_case_metrics(prediction: Any, target: Any, spacing: Sequence[float], tolerance_mm: float) -> dict[str, float]:
    return {"dice": dice_score(prediction, target), **surface_metrics(prediction, target, spacing, tolerance_mm)}
