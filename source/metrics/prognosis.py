from __future__ import annotations

from collections.abc import Sequence

from .calibration import calibration_slope_intercept
from .classification import binary_classification_metrics


def prognosis_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> dict[str, float]:
    return {
        **binary_classification_metrics(y_true, y_prob, threshold),
        **calibration_slope_intercept(y_true, y_prob),
    }
