"""Simplified PESI (sPESI).

Six criteria, one point each; 0 points is low risk and any point is high risk. The
definition and thresholds are the ones the INSPECT release itself scores with --
``third_party/repos/INSPECT_public/ehr/4_compute_pesi_score.py`` -- so a number produced
here is comparable with the clinical baseline reported for that cohort.

A case that is missing any component is deliberately left unscored. sPESI has no
imputation rule, and summing the components that happen to be present would silently
report a low-risk score for a patient whose vitals were never found.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

# The six sPESI criteria. `cardiopulmonary_disease` is one component, not two: the
# criterion is a history of chronic cardiopulmonary disease, which INSPECT resolves from
# a single SNOMED concept rather than separate heart-failure and lung-disease fields.
SPESI_COMPONENTS = (
    "age",
    "cancer",
    "cardiopulmonary_disease",
    "pulse",
    "systolic_bp",
    "oxygen_saturation",
)

DEFAULT_FIELDS = {name: name for name in SPESI_COMPONENTS}

SPESI_FEATURE_COLUMNS = ("spesi", "spesi_high_risk", "spesi_computable")


@dataclass(frozen=True)
class SpesiResult:
    spesi_score: int | None
    high_risk: bool | None
    computable: bool
    missing_components: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, float) and math.isnan(value)


def _number(record: Mapping[str, Any], field: str) -> float | None:
    value = record.get(field)
    return None if _missing(value) else float(value)


def _boolean(record: Mapping[str, Any], field: str) -> bool | None:
    value = record.get(field)
    if _missing(value):
        return None
    if type(value) is bool:
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"cannot parse clinical boolean {field}={value!r}")


def compute_spesi(record: Mapping[str, Any], fields: Mapping[str, str] | None = None) -> SpesiResult:
    """Score one study, or report exactly which components were missing."""
    names = {**DEFAULT_FIELDS, **dict(fields or {})}
    values: dict[str, Any] = {
        "age": _number(record, names["age"]),
        "cancer": _boolean(record, names["cancer"]),
        "cardiopulmonary_disease": _boolean(record, names["cardiopulmonary_disease"]),
        "pulse": _number(record, names["pulse"]),
        "systolic_bp": _number(record, names["systolic_bp"]),
        "oxygen_saturation": _number(record, names["oxygen_saturation"]),
    }
    absent = tuple(name for name in SPESI_COMPONENTS if values[name] is None)
    if absent:
        return SpesiResult(None, None, False, absent)
    if values["age"] < 0:
        raise ValueError("age cannot be negative")
    score = (
        int(values["age"] > 80)
        + int(values["cancer"])
        + int(values["cardiopulmonary_disease"])
        + int(values["pulse"] >= 110)
        + int(values["systolic_bp"] < 100)
        + int(values["oxygen_saturation"] < 90)
    )
    return SpesiResult(score, score >= 1, True, ())


def spesi_score(record: Mapping[str, Any], fields: Mapping[str, str] | None = None) -> int | None:
    return compute_spesi(record, fields).spesi_score
