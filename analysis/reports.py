"""Radiology-report text statistics, and how far the rule extractor reaches.

Reports are the supervision source for silver labels, so knowing their length and how
often the deterministic patterns fire tells you the ceiling of the rule-only arm before
any model is run.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .loaders import DatasetBundle, column_names

TEXT_COLUMNS = ("report_text", "impression", "findings", "text", "report")


def _text_column(rows: Sequence[Mapping[str, Any]]) -> str | None:
    available = set(column_names(rows))
    return next((name for name in TEXT_COLUMNS if name in available), None)


def _describe(values: Sequence[int]) -> dict[str, float] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    count = len(ordered)
    return {
        "n": count,
        "min": ordered[0],
        "p25": ordered[count // 4],
        "median": ordered[count // 2],
        "p75": ordered[(3 * count) // 4],
        "max": ordered[-1],
        "mean": sum(ordered) / count,
    }


def text_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    column = _text_column(rows)
    if column is None:
        return {"status": "unavailable", "reason": "no report text column"}
    characters: list[int] = []
    words: list[int] = []
    empty = 0
    for row in rows:
        text = str(row.get(column) or "").strip()
        if not text:
            empty += 1
            continue
        characters.append(len(text))
        words.append(len(text.split()))
    return {
        "status": "available",
        "column": column,
        "reports": len(rows),
        "empty_reports": empty,
        "empty_rate": empty / len(rows) if rows else None,
        "characters": _describe(characters),
        "words": _describe(words),
    }


def rule_coverage(rows: Sequence[Mapping[str, Any]], maximum: int | None = None) -> dict[str, Any]:
    """How many of the 19 silver targets the regex layer settles, before any LLM runs.

    This is the honest floor for silver coverage: anything the rules cannot resolve has to
    be paid for with an LLM call or become an abstention.
    """
    try:
        from source.silver.rules import RULE_VERSION, apply_rule
        from source.silver.schema import TARGETS
    except Exception as exc:  # noqa: BLE001 - EDA must not fail on an optional import
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    column = _text_column(rows)
    if column is None:
        return {"status": "unavailable", "reason": "no report text column"}
    selected = rows if maximum is None else rows[:maximum]
    resolved: Counter[str] = Counter()
    examined = 0
    for row in selected:
        text = str(row.get(column) or "")
        if not text.strip():
            continue
        examined += 1
        for target in TARGETS:
            try:
                if apply_rule(text, target).resolved:
                    resolved[target] += 1
            except Exception:  # noqa: BLE001 - a bad pattern must not abort the whole pass
                continue
    return {
        "status": "available",
        "rule_version": RULE_VERSION,
        "reports_examined": examined,
        "by_target": {
            target: {
                "resolved": resolved[target],
                "resolved_rate": resolved[target] / examined if examined else None,
            }
            for target in TARGETS
        },
        "mean_targets_resolved_per_report": (
            sum(resolved.values()) / examined if examined else None
        ),
    }


def analyse(bundle: DatasetBundle, rule_sample: int | None = 2000) -> dict[str, Any]:
    rows = bundle.rows("reports.csv") or bundle.rows("paired_reports.csv")
    if not rows:
        return {"status": "unavailable", "reason": "no reports.csv or paired_reports.csv"}
    return {
        "status": "available",
        "text": text_summary(rows),
        "rule_coverage": rule_coverage(rows, rule_sample),
        "rule_coverage_sample": rule_sample,
    }
