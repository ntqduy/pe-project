from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

# Logistic recalibration of outcome on logit(p) is only meaningful with enough cases of
# both outcomes; with a handful of cases and near-constant probabilities the unpenalized
# fit returns slopes in the hundreds (seen: -869 / +805 at n=5-9). Such values are reported
# as unavailable instead of as numbers.
DEFAULT_CALIBRATION_MIN_EVENTS = 10


def _logits(y_prob: Sequence[float]) -> np.ndarray:
    probability = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def calibration_unavailable_reason(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    min_events: int = DEFAULT_CALIBRATION_MIN_EVENTS,
) -> str | None:
    """Why calibration slope/intercept cannot be estimated, or None when they can."""
    truth = np.asarray(y_true, dtype=int)
    if truth.size == 0 or len(np.unique(truth)) != 2:
        return "calibration needs both outcome classes"
    smaller = int(min((truth == 1).sum(), (truth == 0).sum()))
    if smaller < int(min_events):
        return f"calibration needs >= {int(min_events)} cases per class (smaller class = {smaller})"
    logit = _logits(y_prob)
    positive, negative = logit[truth == 1], logit[truth == 0]
    if negative.max() <= positive.min() or positive.max() <= negative.min():
        return "calibration slope not estimable: outcomes perfectly separated by the predictions"
    return None


def calibration_slope_intercept(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    min_events: int = DEFAULT_CALIBRATION_MIN_EVENTS,
) -> dict[str, float]:
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for calibration metrics") from exc
    unavailable = {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    truth = np.asarray(y_true, dtype=int)
    if calibration_unavailable_reason(truth, y_prob, min_events=min_events) is not None:
        return unavailable
    logit = _logits(y_prob).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, truth)
        except ConvergenceWarning:
            return unavailable
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
