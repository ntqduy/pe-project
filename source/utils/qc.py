from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QC_STATUSES = ("pass", "failed", "skipped", "abstained")


def canonical_qc_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    mapping = {
        "pass": "pass",
        "passed": "pass",
        "suspicious": "pass",
        "fail": "failed",
        "failed": "failed",
        "unavailable": "failed",
        "skip": "skipped",
        "skipped": "skipped",
        "cached": "skipped",
        "abstain": "abstained",
        "abstained": "abstained",
        "no_result": "failed",
    }
    if status not in mapping:
        raise ValueError(f"unsupported QC status: {value!r}")
    return mapping[status]


def qc_row(
    *,
    run_id: str,
    patient_id: Any,
    study_id: Any,
    stage: str,
    item_name: Any,
    status: Any,
    input_path: Any = "",
    output_path: Any = "",
    qc_checks: Any = "",
    failure_reason: Any = "",
) -> dict[str, Any]:
    def absolute(value: Any) -> str:
        if value in (None, ""):
            return ""
        return str(Path(str(value)).resolve())

    normalized = canonical_qc_status(status)
    return {
        "run_id": str(run_id),
        "patient_id": str(patient_id or ""),
        "study_id": str(study_id or ""),
        "stage": str(stage),
        "item_name": str(item_name or ""),
        "status": normalized,
        "input_path": absolute(input_path),
        "output_path": absolute(output_path),
        "qc_checks": str(qc_checks or ""),
        # An abstention has a reason too ("confidence below threshold", "providers
        # disagreed"), and the QC file is the place a reviewer looks first. Only pass and
        # skipped are blanked, so no row carries a stale reason from an earlier state.
        "failure_reason": (
            str(failure_reason or "") if normalized in {"failed", "abstained"} else ""
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
