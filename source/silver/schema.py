from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal


TargetKind = Literal["binary", "categorical", "continuous"]
Status = Literal["accepted", "abstained", "no_result"]
SILVER_STATUSES: tuple[Status, ...] = ("accepted", "abstained", "no_result")
# 4 writes the authoritative normalized table as silver_labels.csv. Mixed target values
# remain canonical JSON scalars in the CSV value column and every row carries report_hash.
# 5 drops the falcon_value/verifier_value fields of the removed two-model cascade.
SCHEMA_VERSION = 5
REPORT_HASH_LENGTH = 20


def report_hash(report_id: str) -> str:
    """Stable short digest of a report_id.

    The same token names the per-report state file, so a labels row and the state file it
    was cached in can be matched without the rule being re-derived anywhere else.
    """
    text = str(report_id).strip()
    if not text:
        raise ValueError("report_id cannot be empty")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:REPORT_HASH_LENGTH]


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


_TRUE_WORDS = {"true", "yes", "y", "present", "positive", "seen", "identified", "reported", "1"}
_FALSE_WORDS = {"false", "no", "n", "absent", "negative", "not present", "not seen", "0"}
# "None" is how a Python-minded model writes null, not the answer "no finding".
_NULL_WORDS = {
    "", "null", "none", "none_reported", "unknown", "not mentioned", "not_mentioned", "not stated",
    "not reported", "uncertain", "indeterminate", "n/a", "na", "nan",
}
# A qualitative grade can only describe a finding that is present, so a binary answer such
# as "small" means true (e.g. "SMALL BILATERAL PLEURAL EFFUSIONS"). A side alone does not:
# "normal right ventricle" names the side of a normal structure.
_PRESENCE_WORDS = (
    "trace", "tiny", "minimal", "small", "mild", "moderate", "large", "severe", "massive",
    "loculated", "extensive", "diffuse", "multiple",
)
# Words that keep a grade from meaning presence ("no large effusion", "possible small ...",
# "extensive bilateral emboli, unlikely chronic").
_NOT_PRESENCE_WORDS = {
    "no", "not", "without", "absent", "none", "negative", "normal", "unremarkable", "resolved",
    "possible", "possibly", "probable", "likely", "unlikely", "questionable", "suspected", "may",
    "might", "cannot", "exclude", "excluded", "doubtful", "doubt", "equivocal", "indeterminate",
}
# Targets a grade cannot describe: "small" says nothing about whether a ratio is mentioned
# or abnormal, so such an answer is read as null rather than true.
_UNGRADED_TARGETS = frozenset({"rv_lv_ratio_mentioned", "rv_lv_ratio_abnormal"})
# Trailing sentence punctuation a model sometimes adds to a one-word answer ("No.").
_ANSWER_PUNCTUATION = ".!,;:"
_ACUITY_ALIASES = {
    "acute_on_chronic": "acute_on_chronic", "acute_and_chronic": "acute_on_chronic",
    "acute": "acute", "subacute": "acute", "chronic": "chronic",
    "uncertain": "uncertain", "indeterminate": "uncertain", "age_indeterminate": "uncertain",
    "indeterminate_age": "uncertain", "indeterminate_acuity": "uncertain",
}


def normalize_target_value(target: str, value: Any) -> tuple[bool | str | float | None, str | None]:
    """Coerce a model's near-miss answer to the target type before strict validation.

    Returns ``(value, note)``; ``note`` names the coercion (None when the value was already
    well-typed). Values that cannot be mapped unambiguously are returned unchanged so that
    ``validate_target_value`` still rejects them.
    """
    spec = target_spec(target)
    if value is None:
        return None, None
    if spec.kind == "binary":
        if type(value) is bool:
            return value, None
        if isinstance(value, Real) and float(value) in (0.0, 1.0):
            return bool(value), f"numeric_{value}_as_boolean"
        if isinstance(value, str):
            text = " ".join(value.lower().replace("_", " ").split()).rstrip(_ANSWER_PUNCTUATION).strip()
            if text in _NULL_WORDS or text.replace(" ", "_") in _NULL_WORDS:
                return None, f"text_{value!r}_as_null"
            if text in _TRUE_WORDS:
                return True, f"text_{value!r}_as_true"
            if text in _FALSE_WORDS:
                return False, f"text_{value!r}_as_false"
            words = set(text.replace(",", " ").replace("-", " ").replace(";", " ").split())
            if words & set(_PRESENCE_WORDS) and not words & _NOT_PRESENCE_WORDS:
                if spec.name in _UNGRADED_TARGETS:
                    return None, f"grade_{value!r}_not_applicable_as_null"
                return True, f"grade_{value!r}_as_true"
        return value, None
    if spec.kind == "categorical":
        if isinstance(value, str):
            if value in spec.values:
                return value, None
            key = "_".join(value.strip().lower().replace("-", " ").rstrip(_ANSWER_PUNCTUATION).split())
            # Category names first: "indeterminate" is a valid acuity, not a missing value.
            mapped = _ACUITY_ALIASES.get(key) if spec.name == "acuity" else None
            if mapped is None and key in spec.values:
                mapped = key
            if mapped is not None:
                return mapped, f"text_{value!r}_as_{mapped}"
            if key in _NULL_WORDS or key.replace("_", " ") in _NULL_WORDS:
                return None, f"text_{value!r}_as_null"
        return value, None
    # continuous
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _NULL_WORDS:
            return None, f"text_{value!r}_as_null"
        try:
            return float(text), f"text_{value!r}_as_number"
        except ValueError:
            return value, None
    return value, None


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
    medgemma_value: bool | str | float | None = None

    def __post_init__(self) -> None:
        canonical = canonical_target(self.target)
        object.__setattr__(self, "target", canonical)
        object.__setattr__(self, "value", validate_target_value(canonical, self.value))
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

    @property
    def report_hash(self) -> str:
        """Derived, never stored on the instance, so it cannot drift from report_id."""
        return report_hash(self.report_id)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_storage_dict(self) -> dict[str, Any]:
        """One normalized silver-label row.

        Values stay native JSON scalars (true / "acute" / 1.4 / null). The previous Parquet
        table had to encode them as JSON strings because one Arrow column cannot hold mixed
        boolean, string and float targets; JSON Lines has no such constraint, and
        ``decode_storage_value`` still reads both shapes so older tables remain loadable.
        """
        return {
            "report_hash": self.report_hash,
            **asdict(self),
            "schema_version": SCHEMA_VERSION,
        }


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

