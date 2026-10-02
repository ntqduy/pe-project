"""Eligibility, exclusion and noise-removal rules for the INSPECT cohort.

Generalized from the /mnt/pe_study readiness audits (`audit_study_readiness.py`,
`build_inspect_500_phase1_audit.py`), which bucketed `num_slices`, `SliceThickness`
and `PixelSpacing_0` for review. Here those buckets become explicit, configurable
exclusion rules with a per-study ledger, so every dropped study is attributable.

Two principles:

1. Nothing is dropped silently. Every exclusion is recorded as (study, rule, detail)
   in the ledger that ships next to the manifests.
2. Thresholds are configuration, never constants baked into a stage. Both dataset
   profiles run this identical code with the identical rule block.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sources import StudyRecord

# Applied in this order; the first failing rule owns the exclusion.
RULE_ORDER = (
    "missing_study_id",
    "missing_split",
    "missing_ct_file",
    "missing_report",
    "missing_labels",
    "missing_series_metadata",
    "unsupported_modality",
    "too_few_slices",
    "too_many_slices",
    "slice_thickness_out_of_range",
    "pixel_spacing_out_of_range",
    "missing_ehr_crosswalk",
)

DEFAULT_RULES: dict[str, Any] = {
    "require_ct_file": True,
    "require_report": True,
    "require_labels": True,
    "require_series_metadata": True,
    "require_ehr_crosswalk": False,
    "allowed_modalities": ["CT"],
    "minimum_slices": 2,
    "maximum_slices": None,
    "minimum_slice_thickness_mm": None,
    "maximum_slice_thickness_mm": 5.0,
    "minimum_pixel_spacing_mm": None,
    "maximum_pixel_spacing_mm": 2.0,
}


def _number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


@dataclass
class ExclusionLedger:
    """Every study removed from the cohort, with the rule that removed it."""

    entries: list[dict[str, str]] = field(default_factory=list)

    def record(self, record: StudyRecord, rule: str, detail: str) -> None:
        self.entries.append(
            {
                "patient_id": record.patient_id,
                "study_id": record.study_id,
                "impression_id": record.impression_id,
                "split": record.split,
                "rule": rule,
                "detail": detail,
            }
        )

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(entry["rule"] for entry in self.entries).items()))

    def excluded_studies(self) -> set[str]:
        return {entry["study_id"] for entry in self.entries}


@dataclass(frozen=True)
class EligibilityReport:
    rules: dict[str, Any]
    considered_studies: int
    eligible_studies: int
    eligible_patients: int
    excluded_studies: int
    excluded_by_rule: dict[str, int]
    split_studies: dict[str, int]
    split_patients: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rules": self.rules,
            "considered_studies": self.considered_studies,
            "eligible_studies": self.eligible_studies,
            "eligible_patients": self.eligible_patients,
            "excluded_studies": self.excluded_studies,
            "excluded_by_rule": self.excluded_by_rule,
            "split_studies": self.split_studies,
            "split_patients": self.split_patients,
        }


def resolve_rules(rules: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = dict(DEFAULT_RULES)
    unknown = sorted(set(rules or {}) - set(DEFAULT_RULES))
    if unknown:
        raise ValueError("unknown eligibility rule(s): " + ", ".join(unknown))
    resolved.update(dict(rules or {}))
    return resolved


def _evaluate(record: StudyRecord, rules: Mapping[str, Any], *, check_files: bool) -> tuple[str, str] | None:
    """Return (rule, detail) for the first rule this study fails, or None if eligible."""
    if not record.study_id:
        return "missing_study_id", "study has no image_id in study_mapping"
    if record.split not in {"train", "validation", "test"}:
        return "missing_split", f"split={record.split!r}"
    if rules["require_ct_file"] and check_files:
        path = Path(record.image_path) if record.image_path else None
        if path is None or not path.is_file():
            return "missing_ct_file", str(path or "no image path")
    if rules["require_report"] and not record.report_text:
        return "missing_report", "impression text is empty"
    if rules["require_labels"] and not record.labels:
        return "missing_labels", "no row in the official labels table"
    if rules["require_series_metadata"] and not any(record.series.values()):
        return "missing_series_metadata", "no row in the official series metadata table"

    allowed = [str(value).strip().upper() for value in (rules["allowed_modalities"] or [])]
    modality = str(record.study.get("Modality") or "").strip().upper()
    if allowed and modality and modality not in allowed:
        return "unsupported_modality", f"Modality={modality}"

    slices = _number(record.series.get("num_slices"))
    if rules["minimum_slices"] is not None and slices is not None and slices < float(rules["minimum_slices"]):
        return "too_few_slices", f"num_slices={slices:g}"
    if rules["maximum_slices"] is not None and slices is not None and slices > float(rules["maximum_slices"]):
        return "too_many_slices", f"num_slices={slices:g}"

    thickness = _number(record.series.get("SliceThickness"))
    if thickness is not None:
        low = rules["minimum_slice_thickness_mm"]
        high = rules["maximum_slice_thickness_mm"]
        if (low is not None and thickness < float(low)) or (high is not None and thickness > float(high)):
            return "slice_thickness_out_of_range", f"SliceThickness={thickness:g}"

    spacings = [_number(record.series.get(name)) for name in ("PixelSpacing_0", "PixelSpacing_1")]
    for name, spacing in zip(("PixelSpacing_0", "PixelSpacing_1"), spacings):
        if spacing is None:
            continue
        low = rules["minimum_pixel_spacing_mm"]
        high = rules["maximum_pixel_spacing_mm"]
        if (low is not None and spacing < float(low)) or (high is not None and spacing > float(high)):
            return "pixel_spacing_out_of_range", f"{name}={spacing:g}"

    if rules["require_ehr_crosswalk"] and not record.has_ehr_crosswalk:
        return "missing_ehr_crosswalk", "no image/EHR crosswalk row"
    return None


def apply_eligibility(
    records: Sequence[StudyRecord],
    rules: Mapping[str, Any] | None = None,
    *,
    excluded_patients: Sequence[str] | set[str] = (),
    check_files: bool = True,
    ledger: ExclusionLedger | None = None,
) -> tuple[list[StudyRecord], ExclusionLedger, EligibilityReport]:
    """Filter studies to the eligible cohort shared by every dataset profile.

    ``excluded_patients`` removes whole patients (governance / previously-seen
    registries) before any study-level rule runs, so a patient is never partially
    present. ``check_files`` may be disabled for metadata-only static validation.
    """
    resolved = resolve_rules(rules)
    ledger = ledger or ExclusionLedger()
    blocked = {str(value).strip() for value in excluded_patients if str(value).strip()}
    eligible: list[StudyRecord] = []
    for record in records:
        if record.patient_id in blocked:
            ledger.record(record, "excluded_patient_registry", "patient is on the exclusion registry")
            continue
        failure = _evaluate(record, resolved, check_files=check_files)
        if failure is None:
            eligible.append(record)
        else:
            ledger.record(record, failure[0], failure[1])

    split_studies = Counter(record.split for record in eligible)
    split_patients: dict[str, set[str]] = {}
    for record in eligible:
        split_patients.setdefault(record.split, set()).add(record.patient_id)
    report = EligibilityReport(
        rules=resolved,
        considered_studies=len(records),
        eligible_studies=len(eligible),
        eligible_patients=len({record.patient_id for record in eligible}),
        excluded_studies=len(ledger.entries),
        excluded_by_rule=ledger.counts(),
        split_studies=dict(sorted(split_studies.items())),
        split_patients={key: len(value) for key, value in sorted(split_patients.items())},
    )
    return eligible, ledger, report
