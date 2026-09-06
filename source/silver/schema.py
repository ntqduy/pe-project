from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal, Mapping, Sequence


TargetKind = Literal["binary", "categorical", "continuous"]
Status = Literal["accepted", "abstained", "no_result"]
SILVER_STATUSES: tuple[Status, ...] = ("accepted", "abstained", "no_result")
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: TargetKind
    description: str
    values: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @property
    def allowed_prompt_values(self) -> str:
        if self.kind == "binary":
            return "[true, false, null]"
        if self.kind == "categorical":
            return json.dumps([*self.values, None], separators=(",", ":"))
        return "a finite JSON number or null"


_SPECS = (
    TargetSpec("pe_present", "binary", "Definite pulmonary embolism present on this examination."),
    TargetSpec(
        "acuity",
        "categorical",
        "Reported pulmonary embolism acuity.",
        values=("acute", "chronic", "acute_on_chronic", "uncertain"),
    ),
    TargetSpec("central", "binary", "Pulmonary embolus in the pulmonary trunk or main pulmonary arteries."),
    TargetSpec("lobar", "binary", "Pulmonary embolus at lobar arterial level."),
    TargetSpec("segmental", "binary", "Pulmonary embolus at segmental arterial level."),
    TargetSpec("subsegmental", "binary", "Pulmonary embolus at subsegmental arterial level."),
    TargetSpec("saddle", "binary", "Saddle pulmonary embolus spanning the main pulmonary artery bifurcation."),
    TargetSpec("rv_enlargement", "binary", "Right-ventricular enlargement is explicitly reported."),
    TargetSpec(
        "rv_lv_ratio_mentioned",
        "binary",
        "An RV/LV ratio is explicitly mentioned, whether or not a reliable numeric value is given.",
    ),
    TargetSpec(
        "rv_lv_ratio_value",
        "continuous",
        "Reliable numeric RV/LV ratio stated in the report.",
        minimum=0.0,
    ),
    TargetSpec(
        "rv_lv_ratio_abnormal",
        "binary",
        "The report explicitly calls the RV/LV ratio elevated or abnormal.",
        aliases=("rv_lv_abnormal",),
    ),
    TargetSpec("septal_bowing", "binary", "Interventricular septal bowing is explicitly reported."),
    TargetSpec(
        "contrast_reflux",
        "binary",
        "Contrast reflux into the IVC or hepatic veins is explicitly reported.",
        aliases=("reflux",),
    ),
    TargetSpec("pleural_effusion", "binary", "Pleural effusion is explicitly reported."),
    TargetSpec("pericardial_effusion", "binary", "Pericardial effusion is explicitly reported."),
    TargetSpec(
        "malignancy_related_finding",
        "binary",
        "A finding explicitly described as malignant, metastatic, or suspicious for malignancy.",
    ),
    TargetSpec("chronic_lung_disease", "binary", "Chronic lung disease is explicitly reported."),
    TargetSpec("fibrosis", "binary", "Pulmonary fibrosis or fibrotic lung disease is explicitly reported."),
    TargetSpec("emphysema", "binary", "Pulmonary emphysema is explicitly reported."),
)

TARGET_SPECS: dict[str, TargetSpec] = {spec.name: spec for spec in _SPECS}
TARGETS = tuple(TARGET_SPECS)
ACUITY_VALUES = TARGET_SPECS["acuity"].values
TARGET_ALIASES: dict[str, str] = {
    alias: spec.name for spec in _SPECS for alias in spec.aliases
}


def canonical_target(target: str) -> str:
    normalized = str(target).strip()
    canonical = TARGET_ALIASES.get(normalized, normalized)
    if canonical not in TARGET_SPECS:
        raise ValueError(f"unknown silver target: {target}")
    return canonical


def target_spec(target: str) -> TargetSpec:
    return TARGET_SPECS[canonical_target(target)]


def _validate_confidence(confidence: Any) -> float | None:
    if confidence is None:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("silver confidence must be a number in [0, 1] or null")
    numeric = float(confidence)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("silver confidence must be finite and in [0, 1]")
    return numeric


def validate_target_value(target: str, value: Any) -> bool | str | float | None:
    spec = target_spec(target)
    if value is None:
        return None
    if spec.kind == "binary":
        if type(value) is not bool:
            raise ValueError(f"{spec.name} must be true, false, or null")
        return value
    if spec.kind == "categorical":
        if not isinstance(value, str) or value not in spec.values:
            raise ValueError(f"invalid {spec.name}: {value!r}; expected one of {spec.values}")
        return value
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{spec.name} must be a finite number or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{spec.name} must be finite")
    if spec.minimum is not None and numeric < spec.minimum:
        raise ValueError(f"{spec.name} must be >= {spec.minimum}")
    if spec.maximum is not None and numeric > spec.maximum:
        raise ValueError(f"{spec.name} must be <= {spec.maximum}")
    return numeric


@dataclass(frozen=True)
class SilverLabel:
    patient_id: str
    study_id: str
    report_id: str
    target: str
    value: bool | str | float | None
    status: Status
    source: str
    confidence: float | None = None
    provider: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    reason: str | None = None
    rule_version: str | None = None
    falcon_value: bool | str | float | None = None
    verifier_value: bool | str | float | None = None
    medgemma_value: bool | str | float | None = None

    def __post_init__(self) -> None:
        canonical = canonical_target(self.target)
        object.__setattr__(self, "target", canonical)
        object.__setattr__(self, "value", validate_target_value(canonical, self.value))
        object.__setattr__(self, "falcon_value", validate_target_value(canonical, self.falcon_value))
        object.__setattr__(self, "verifier_value", validate_target_value(canonical, self.verifier_value))
        object.__setattr__(self, "medgemma_value", validate_target_value(canonical, self.medgemma_value))
        if self.status not in SILVER_STATUSES:
            raise ValueError(f"unsupported silver status: {self.status!r}")
        if self.status == "accepted" and self.value is None:
            raise ValueError("accepted silver label cannot have null value")
        if self.status != "accepted" and self.value is not None:
            raise ValueError("abstained/no_result silver label must preserve null final value")
        if not self.patient_id.strip() or not self.study_id.strip() or not self.report_id.strip():
            raise ValueError("silver label identifiers cannot be empty")
        if not str(self.source).strip():
            raise ValueError("silver label source cannot be empty")
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))
        for name in ("provider", "model_id", "model_revision", "reason", "rule_version"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_storage_dict(self) -> dict[str, Any]:
        """Arrow-compatible normalized row; mixed scalar values use JSON scalars."""
        payload = asdict(self)
        for field in ("value", "falcon_value", "verifier_value", "medgemma_value"):
            value = payload[field]
            payload[field] = json.dumps(value, separators=(",", ":")) if value is not None else None
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    @classmethod
    def from_storage_dict(cls, row: Mapping[str, Any]) -> "SilverLabel":
        required = {"patient_id", "study_id", "report_id", "target", "value", "status", "source"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError("silver row missing required fields: " + ", ".join(missing))
        kwargs = {field: row.get(field) for field in cls.__dataclass_fields__}
        for field in ("value", "falcon_value", "verifier_value", "medgemma_value"):
            kwargs[field] = decode_storage_value(kwargs.get(field))
        return cls(**kwargs)


def decode_storage_value(value: Any) -> bool | str | float | None:
    if value is None:
        return None
    if type(value) is bool:
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = text
    elif isinstance(value, Real) and not isinstance(value, bool):
        decoded = float(value)
    else:
        raise ValueError(f"invalid stored silver value: {value!r}")
    if decoded is None or type(decoded) is bool or isinstance(decoded, str):
        return decoded
    if isinstance(decoded, Real) and not isinstance(decoded, bool):
        numeric = float(decoded)
        if math.isfinite(numeric):
            return numeric
    raise ValueError(f"invalid stored silver value: {value!r}")


def validate_label_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_report_ids: Sequence[str] | None = None,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Strictly validate and canonicalize normalized report-target rows."""

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    identity_by_report: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        try:
            label = SilverLabel.from_storage_dict(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid silver row {index}: {exc}") from exc
        key = (label.report_id, label.target)
        if key in seen:
            raise ValueError(f"duplicate silver report-target row: {key}")
        seen.add(key)
        identity = (label.patient_id, label.study_id)
        previous = identity_by_report.setdefault(label.report_id, identity)
        if previous != identity:
            raise ValueError(f"report_id maps to multiple patient/study identities: {label.report_id}")
        normalized.append(label.as_storage_dict())
    if expected_report_ids is not None:
        expected = tuple(str(value).strip() for value in expected_report_ids)
        if not all(expected) or len(expected) != len(set(expected)):
            raise ValueError("expected report IDs must be non-empty and unique")
        actual_reports = set(identity_by_report)
        if actual_reports != set(expected):
            raise ValueError(
                "silver report coverage mismatch: "
                f"missing={sorted(set(expected)-actual_reports)} extra={sorted(actual_reports-set(expected))}"
            )
        if require_complete:
            expected_keys = {(report_id, target) for report_id in expected for target in TARGETS}
            if seen != expected_keys:
                raise ValueError(
                    "silver target coverage mismatch: "
                    f"missing={len(expected_keys-seen)} extra={len(seen-expected_keys)}"
                )
    return normalized


def wide_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert validated normalized rows into one compact row per report/study."""

    normalized = validate_label_rows(rows, require_complete=False)
    grouped: dict[str, dict[str, Any]] = {}
    present: dict[str, set[str]] = {}
    for row in normalized:
        report_id = str(row["report_id"])
        base = grouped.setdefault(
            report_id,
            {
                "schema_version": SCHEMA_VERSION,
                "patient_id": str(row["patient_id"]),
                "study_id": str(row["study_id"]),
                "report_id": report_id,
            },
        )
        target = str(row["target"])
        present.setdefault(report_id, set()).add(target)
        value = decode_storage_value(row.get("value"))
        base[f"{target}__value"] = value
        base[f"{target}__status"] = row.get("status")
        base[f"{target}__confidence"] = row.get("confidence")
        base[f"{target}__source"] = row.get("source")
        base[f"{target}__provider"] = row.get("provider")
    for report_id, observed in present.items():
        missing = set(TARGETS) - observed
        if missing:
            raise ValueError(f"report {report_id} is missing targets: {sorted(missing)}")
    return [grouped[key] for key in sorted(grouped)]
