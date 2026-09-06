from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def calibration_slope_intercept(y_true: Sequence[int], y_prob: Sequence[float]) -> dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for calibration metrics") from exc
    truth = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(truth)) != 2:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, truth)
    return {"calibration_intercept": float(model.intercept_[0]), "calibration_slope": float(model.coef_[0, 0])}


def calibration_curve_points(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    bins: int = 10,
    strategy: str = "quantile",
) -> list[dict[str, float | int]]:
    if bins < 2:
        raise ValueError("calibration curve requires at least two bins")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("calibration strategy must be uniform or quantile")
    try:
        from sklearn.calibration import calibration_curve
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for calibration curves") from exc
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_prob, dtype=float)
    observed, predicted = calibration_curve(truth, probability, n_bins=bins, strategy=strategy)
    return [
        {
            "bin": index,
            "mean_predicted_probability": float(expected),
            "observed_event_fraction": float(actual),
        }
        for index, (expected, actual) in enumerate(zip(predicted, observed))
    ]
