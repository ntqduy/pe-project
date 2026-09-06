from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(frozen=True)
class AuditRecord:
    """Full decision trail for one report/target silver-label decision.

    Kept out of the compact ``labels.parquet`` training table (which stays a small,
    training-convenient long-form table) and written to a separate ``audit.jsonl``
    instead, so free-text evidence and every provider's raw output remain inspectable
    without slowing down training reads.
    """

    patient_id: str
    study_id: str
    report_id: str
    target: str
    final_status: str
    final_value: bool | str | float | None
    final_source: str
    rule_output: dict[str, Any] | None
    falcon_output: dict[str, Any] | None
    medgemma_output: dict[str, Any] | None
    evidence_text: str | None
    evidence_start: int | None
    evidence_end: int | None
    provider: str | None
    model_id: str | None
    model_revision: str | None
    prompt_version: str | None
    run_id: str
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evidence_from_outputs(
    falcon_output: Mapping[str, Any] | None,
    medgemma_output: Mapping[str, Any] | None,
) -> tuple[str | None, int | None, int | None]:
    """First non-empty evidence span across the provider outputs that produced a value."""
    for payload in (medgemma_output, falcon_output):
        if payload and payload.get("evidence_text"):
            return payload.get("evidence_text"), payload.get("evidence_start"), payload.get("evidence_end")
    return None, None, None
