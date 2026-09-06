from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PESIResult:
    pesi_score: int | None
    pesi_class: int | None
    spesi_score: int | None
    pesi_computable: bool
    spesi_computable: bool

    def as_dict(self) -> dict[str, int | bool | None]:
        return asdict(self)


DEFAULT_FIELDS = {
    "age": "age",
    "male": "male",
    "cancer": "cancer",
    "heart_failure": "heart_failure",
    "chronic_lung_disease": "chronic_lung_disease",
    "pulse": "pulse",
    "systolic_bp": "systolic_bp",
    "respiratory_rate": "respiratory_rate",
    "temperature_c": "temperature_c",
    "altered_mental_status": "altered_mental_status",
    "oxygen_saturation": "oxygen_saturation",
}


def _missing(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def _number(record: Mapping[str, Any], field: str) -> float | None:
    value = record.get(field)
    return None if _missing(value) else float(value)


def _boolean(record: Mapping[str, Any], field: str) -> bool | None:
    value = record.get(field)
    if _missing(value):
        return None
    if type(value) is bool:
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "male", "m"}:
        return True
    if text in {"false", "no", "n", "0", "female", "f"}:
        return False
    raise ValueError(f"cannot parse clinical boolean {field}={value!r}")


def _pesi_class(score: int) -> int:
    if score <= 65:
        return 1
    if score <= 85:
        return 2
    if score <= 105:
        return 3
    if score <= 125:
        return 4
    return 5


def calculate_pesi(record: Mapping[str, Any], fields: Mapping[str, str] | None = None) -> tuple[int | None, int | None]:
    names = {**DEFAULT_FIELDS, **dict(fields or {})}
    age = _number(record, names["age"])
    male = _boolean(record, names["male"])
    cancer = _boolean(record, names["cancer"])
    heart_failure = _boolean(record, names["heart_failure"])
    chronic_lung = _boolean(record, names["chronic_lung_disease"])
    pulse = _number(record, names["pulse"])
    systolic = _number(record, names["systolic_bp"])
    respiration = _number(record, names["respiratory_rate"])
    temperature = _number(record, names["temperature_c"])
    mental = _boolean(record, names["altered_mental_status"])
    oxygen = _number(record, names["oxygen_saturation"])
    values = (age, male, cancer, heart_failure, chronic_lung, pulse, systolic, respiration, temperature, mental, oxygen)
    if any(value is None for value in values):
        return None, None
    if age < 0:
        raise ValueError("age cannot be negative")
    score = int(age)
    score += 10 if male else 0
    score += 30 if cancer else 0
    score += 10 if heart_failure else 0
    score += 10 if chronic_lung else 0
    score += 20 if pulse >= 110 else 0
    score += 30 if systolic < 100 else 0
    score += 20 if respiration >= 30 else 0
    score += 20 if temperature < 36 else 0
    score += 60 if mental else 0
    score += 20 if oxygen < 90 else 0
    return score, _pesi_class(score)


def calculate_spesi(record: Mapping[str, Any], fields: Mapping[str, str] | None = None) -> int | None:
    names = {**DEFAULT_FIELDS, **dict(fields or {})}
    age = _number(record, names["age"])
    cancer = _boolean(record, names["cancer"])
    heart_failure = _boolean(record, names["heart_failure"])
    chronic_lung = _boolean(record, names["chronic_lung_disease"])
    pulse = _number(record, names["pulse"])
    systolic = _number(record, names["systolic_bp"])
    oxygen = _number(record, names["oxygen_saturation"])
    values = (age, cancer, heart_failure, chronic_lung, pulse, systolic, oxygen)
    if any(value is None for value in values):
        return None
    return int(age > 80) + int(cancer) + int(heart_failure or chronic_lung) + int(pulse >= 110) + int(systolic < 100) + int(oxygen < 90)


def compute_pesi(record: Mapping[str, Any], fields: Mapping[str, str] | None = None) -> PESIResult:
    score, risk_class = calculate_pesi(record, fields)
    simplified = calculate_spesi(record, fields)
    return PESIResult(
        pesi_score=score,
        pesi_class=risk_class,
        spesi_score=simplified,
        pesi_computable=score is not None,
        spesi_computable=simplified is not None,
    )


def compute_spesi(record: Mapping[str, Any], fields: Mapping[str, str] | None = None) -> int | None:
    return calculate_spesi(record, fields)
