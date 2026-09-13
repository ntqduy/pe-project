"""Label prevalence, missingness and class imbalance, per split."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .loaders import DatasetBundle, column_names

# The three reviewed native diagnosis labels the protocol matrix trains on, plus the
# historical aliases the manifest may still carry.
DIAGNOSIS_LABELS = (
    "pe_positive", "pe_acute", "pe_subsegmental",
    "pe_present", "pe_positive_nlp", "pe_subsegmental_only", "pe_subsegmentalonly",
)
MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "unknown", "censored"}


def _state(value: Any) -> str:
    """Three outcomes only: positive, negative, missing. Empty is never a zero."""
    text = str(value).strip().lower() if value is not None else ""
    if text in MISSING_TOKENS:
        return "missing"
    if text in {"1", "1.0", "true", "yes", "t", "positive"}:
        return "positive"
    if text in {"0", "0.0", "false", "no", "f", "negative"}:
        return "negative"
    return "other"


def label_distribution(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> dict[str, Any]:
    total = len(rows)
    output: dict[str, Any] = {}
    for column in columns:
        counts: Counter[str] = Counter(_state(row.get(column)) for row in rows)
        observed = counts["positive"] + counts["negative"]
        output[column] = {
            "rows": total,
            "positive": counts["positive"],
            "negative": counts["negative"],
            "missing": counts["missing"],
            "other": counts["other"],
            "observed": observed,
            # Prevalence is over OBSERVED rows, not over all rows: a missing label is not a
            # negative, and dividing by `total` would silently deflate every prevalence.
            "prevalence_of_observed": counts["positive"] / observed if observed else None,
            "missing_rate": counts["missing"] / total if total else None,
            "imbalance_ratio": (
                counts["negative"] / counts["positive"] if counts["positive"] else None
            ),
        }
    return output


def label_distribution_by_split(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("split") or "unknown").strip().lower()].append(row)
    return {
        split: label_distribution(split_rows, columns)
        for split, split_rows in sorted(grouped.items())
    }


def detect_label_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    available = set(column_names(rows))
    return [name for name in DIAGNOSIS_LABELS if name in available]


def analyse(bundle: DatasetBundle) -> dict[str, Any]:
    rows = bundle.rows("diagnosis.csv") or bundle.rows("ctpa.csv")
    if not rows:
        return {"status": "unavailable", "reason": "no diagnosis.csv or ctpa.csv"}
    columns = detect_label_columns(rows)
    if not columns:
        return {"status": "unavailable", "reason": "no recognised diagnosis label column"}
    return {
        "status": "available",
        "label_columns": columns,
        "overall": label_distribution(rows, columns),
        "by_split": label_distribution_by_split(rows, columns),
    }
