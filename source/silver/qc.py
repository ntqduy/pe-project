from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .schema import TARGETS, decode_storage_value, validate_target_value

PREVIEW_CHARS = 240
PREVIEW_SAMPLES_PER_STATUS = 3
QC_RECORD_SCHEMA_VERSION = 1
QC_ISSUES = ("conflict", "low_conf", "missing_field", "impossible_value")


def _impossible_value(row: Mapping[str, Any]) -> str | None:
    """Reason an accepted row's value is not legal for its target, or None when it is."""
    if str(row.get("status")) != "accepted":
        return None
    try:
        validate_target_value(str(row["target"]), decode_storage_value(row.get("value")))
    except ValueError as exc:
        return str(exc)
    return None


def _impossible_values(labels: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"{row.get('report_id')}/{row.get('target')}: {problem}"
        for row in labels
        if (problem := _impossible_value(row)) is not None
    ]


def _issue(row: Mapping[str, Any]) -> str | None:
    """Classify one label row, or None when nothing is wrong with it.

    Ordered most specific first. Anything non-accepted that is not an explicit
    disagreement or an explicit uncertainty means no source found the finding stated in
    the report, which is what missing_field records.
    """
    status = str(row.get("status") or "")
    reason = str(row.get("reason") or "")
    if status == "accepted":
        return "impossible_value" if _impossible_value(row) else None
    if "disagree" in reason:
        return "conflict"
    if status == "no_result":
        return "missing_field"
    if reason.endswith("_uncertain"):
        return "low_conf"
    return "missing_field"


def silver_qc_records(
    labels: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """One record per problematic (report, target); accepted and valid rows are omitted.

    This is the review queue, not a statistics table: the aggregate counts live in
    result.json, so repeating them here would only create a second thing to keep in sync.
    """
    records: list[dict[str, Any]] = []
    for row in labels:
        issue = _issue(row)
        if issue is None:
            continue
        records.append(
            {
                "schema_version": QC_RECORD_SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "issue": issue,
                "detail": _impossible_value(row) if issue == "impossible_value" else str(row.get("reason") or ""),
                "report_hash": row.get("report_hash"),
                "patient_id": row.get("patient_id"),
                "study_id": row.get("study_id"),
                "report_id": row.get("report_id"),
                "target": row.get("target"),
                "status": row.get("status"),
                "source": row.get("source"),
                "provider": row.get("provider"),
                "confidence": row.get("confidence"),
                "falcon_value": row.get("falcon_value"),
                "medgemma_value": row.get("medgemma_value"),
            }
        )
    return records


def _provider_disagreement(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    both_ran = 0
    disagreed = 0
    per_target: Counter[str] = Counter()
    for record in audits:
        falcon_output = record.get("falcon_output")
        medgemma_output = record.get("medgemma_output")
        if not falcon_output or not medgemma_output:
            continue
        falcon_value = falcon_output.get("value")
        medgemma_value = medgemma_output.get("value")
        if falcon_value is None or medgemma_value is None:
            continue
        both_ran += 1
        if falcon_value != medgemma_value:
            disagreed += 1
            per_target[str(record.get("target"))] += 1
    return {
        "both_providers_ran": both_ran,
        "disagreed": disagreed,
        "disagreement_rate": disagreed / both_ran if both_ran else 0.0,
        "by_target": dict(sorted(per_target.items())),
    }


def _missingness(labels: Sequence[Mapping[str, Any]], report_count: int) -> dict[str, float]:
    accepted_by_target: Counter[str] = Counter()
    for row in labels:
        if str(row.get("status")) == "accepted":
            accepted_by_target[str(row.get("target"))] += 1
    return {
        target: 1.0 - (accepted_by_target[target] / report_count if report_count else 0.0)
        for target in TARGETS
    }


def _sample_previews(
    labels: Sequence[Mapping[str, Any]],
    reports: Mapping[str, str] | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        status = str(row.get("status"))
        if len(grouped[status]) >= PREVIEW_SAMPLES_PER_STATUS:
            continue
        report_id = str(row.get("report_id"))
        preview = None
        if reports is not None:
            text = reports.get(report_id)
            if text:
                preview = text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")
        grouped[status].append(
            {
                "report_id": report_id,
                "target": row.get("target"),
                "reason": row.get("reason"),
                "report_preview": preview,
            }
        )
    return dict(grouped)


def silver_qc_summary(
    labels: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    report_count: int,
    *,
    reports: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the silver-generation QC artifact: prevalence, disposition rates,
    provider disagreement, impossible-value checks, missingness, and report previews."""
    statuses = Counter(str(row["status"]) for row in labels)
    source_status = Counter((str(row["source"]), str(row["status"])) for row in labels)
    target_status = Counter((str(row["target"]), str(row["status"])) for row in labels)
    reasons = Counter(str(row.get("reason") or "unspecified") for row in labels)
    report_statuses: dict[str, set[str]] = defaultdict(set)
    for row in labels:
        report_statuses[str(row["report_id"])].add(str(row["status"]))
    total = len(labels)
    return {
        "reports": report_count,
        "targets": total,
        "accepted": statuses["accepted"],
        "abstained": statuses["abstained"],
        "no_result": statuses["no_result"],
        "coverage": statuses["accepted"] / total if total else 0.0,
        "abstention_rate": statuses["abstained"] / total if total else 0.0,
        "no_result_rate": statuses["no_result"] / total if total else 0.0,
        "reports_with_abstention": sum("abstained" in values for values in report_statuses.values()),
        "reports_with_no_result": sum("no_result" in values for values in report_statuses.values()),
        "by_target": {
            target: {
                status: target_status[(target, status)]
                for status in ("accepted", "abstained", "no_result")
            }
            for target in TARGETS
        },
        "by_source_and_status": {
            source: {
                status: source_status[(source, status)]
                for status in ("accepted", "abstained", "no_result")
            }
            for source in sorted({str(row["source"]) for row in labels})
        },
        "reason_counts": dict(sorted(reasons.items())),
        # Totals for what silver_label_confidence.csv records case by case, so result.json
        # alone says how big the review queue is without that file having to be read.
        "issue_counts": {
            issue: sum(1 for row in labels if _issue(row) == issue) for issue in QC_ISSUES
        },
        "missingness_by_target": _missingness(labels, report_count),
        "provider_disagreement": _provider_disagreement(audits),
        "impossible_values": _impossible_values(labels),
        "sample_report_previews": _sample_previews(labels, reports),
        "expert_review_queue": False,
    }
