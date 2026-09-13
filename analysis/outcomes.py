"""Prognosis endpoints: event counts, censoring and events-per-split.

Event count is the number that decides the evaluation protocol: a small event count is
exactly the case where the study plan switches from a single hold-out to repeated or
nested patient-level cross-validation.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .labels import _state
from .loaders import DatasetBundle, column_names

ENDPOINTS = (
    "1_month_mortality",
    "6_month_mortality",
    "12_month_mortality",
    "1_month_readmission",
    "6_month_readmission",
    "12_month_readmission",
    "12_month_PH",
    "mortality_30d",
)
# Rule of thumb used across the prognostic-modelling literature: roughly ten outcome
# events per predictor. Below this, a single hold-out split is unreliable.
EVENTS_PER_VARIABLE_FLOOR = 10


def endpoint_summary(rows: Sequence[Mapping[str, Any]], endpoint: str) -> dict[str, Any]:
    states = [_state(row.get(endpoint)) for row in rows]
    events = states.count("positive")
    non_events = states.count("negative")
    missing = states.count("missing")
    observed = events + non_events
    censored_column = f"is_censored_{endpoint.split('_')[-1]}"
    censored = sum(1 for row in rows if _state(row.get(censored_column)) == "positive")
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: {"events": 0, "observed": 0})
    for row, state in zip(rows, states):
        split = str(row.get("split") or "unknown").strip().lower()
        if state in {"positive", "negative"}:
            by_split[split]["observed"] += 1
        if state == "positive":
            by_split[split]["events"] += 1
    return {
        "events": events,
        "non_events": non_events,
        "missing": missing,
        "observed": observed,
        "event_rate_of_observed": events / observed if observed else None,
        "censored_rows": censored,
        "by_split": {split: dict(values) for split, values in sorted(by_split.items())},
        "minimum_split_events": min(
            (values["events"] for values in by_split.values()), default=0
        ),
        # Advisory only: the protocol decision is the author's, this just flags the case.
        "single_holdout_advisable": events >= EVENTS_PER_VARIABLE_FLOOR * 10,
        "recommended_protocol": (
            "patient_holdout"
            if events >= EVENTS_PER_VARIABLE_FLOOR * 10
            else "repeated or nested patient-level CV (small event count)"
        ),
    }


def analyse(bundle: DatasetBundle) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("prognosis_pe_positive.csv", "prognosis_all_patient.csv", "prognosis.csv"):
        rows = bundle.rows(name)
        if not rows:
            continue
        available = set(column_names(rows))
        endpoints = [endpoint for endpoint in ENDPOINTS if endpoint in available]
        if not endpoints:
            output[name] = {"status": "unavailable", "reason": "no recognised endpoint column"}
            continue
        output[name] = {
            "status": "available",
            "rows": len(rows),
            "endpoints": {
                endpoint: endpoint_summary(rows, endpoint) for endpoint in endpoints
            },
        }
    return output or {"status": "unavailable", "reason": "no prognosis manifest"}
