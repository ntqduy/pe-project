"""Study-level PE labels used only to annotate segmentation QC previews."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def parse_pe_present(value: Any) -> bool | None:
    """Preserve missing labels instead of treating them as PE-negative."""
    if value is None or str(value).strip() == "":
        return None
    if value is True or str(value).strip().lower() in {"1", "true"}:
        return True
    if value is False or str(value).strip().lower() in {"0", "false"}:
        return False
    raise ValueError(f"invalid pe_present label: {value!r}")


def study_pe_labels(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], bool | None]:
    """Index the diagnosis manifest by patient and study, rejecting conflicting labels."""
    labels: dict[tuple[str, str], bool | None] = {}
    for row in rows:
        key = (str(row.get("patient_id") or "").strip(), str(row.get("study_id") or "").strip())
        if not all(key):
            raise ValueError("diagnosis rows require patient_id and study_id")
        label = parse_pe_present(row.get("pe_present"))
        if key in labels and labels[key] != label:
            raise ValueError(f"conflicting pe_present labels for patient/study {key}")
        labels[key] = label
    return labels
