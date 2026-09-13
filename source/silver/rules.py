from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Pattern

from .schema import canonical_target


RULE_VERSION = "pe_rules_v2"


@dataclass(frozen=True)
class EvidenceSpan:
    text: str
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleDecision:
    resolved: bool
    value: bool | str | float | None
    reason: str
    evidence: EvidenceSpan | None = None
    version: str = RULE_VERSION

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


_FLAGS = re.IGNORECASE
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_UNCERTAIN = re.compile(
    r"\b(cannot exclude|may represent|might represent|possible|possibly|question of|"
    r"equivocal|indeterminate|suspicious for|concerning for)\b",
    _FLAGS,
)
_HISTORICAL = re.compile(r"\b(history of|previously seen|previously noted|remote history of)\b", _FLAGS)


def _patterns(*values: str) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(value, _FLAGS) for value in values)


_PE_TERM = r"pulmonary embol(?:us|i|ism)"
_PE_NEGATIVE = _patterns(
    rf"\b(?:no|without)\s+(?:evidence of\s+)?(?:acute\s+|chronic\s+)?{_PE_TERM}\b",
    rf"\bnegative\s+for\s+(?:acute\s+|chronic\s+)?{_PE_TERM}\b",
    r"\bno\s+(?:intraluminal\s+)?filling defects?[^.;\n]{0,45}\bpulmonary arter(?:y|ies)\b",
)
_PE_POSITIVE = _patterns(rf"\b(?:acute\s+|chronic\s+)?{_PE_TERM}\b")


_BOOLEAN_PATTERNS: dict[str, tuple[tuple[Pattern[str], ...], tuple[Pattern[str], ...]]] = {
    "saddle": (
        _patterns(r"\bsaddle\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b"),
        _patterns(r"\b(?:no|without)\s+saddle\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b"),
    ),
    "central": (
        _patterns(
            r"\bcentral\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b",
            r"\b(?:embolus|embolism|thrombus|filling defect)\b[^.;\n]{0,50}"
            r"\b(?:main pulmonary arter(?:y|ies)|pulmonary trunk)\b",
            r"\b(?:main pulmonary arter(?:y|ies)|pulmonary trunk)\b[^.;\n]{0,50}"
            r"\b(?:embolus|embolism|thrombus|filling defect)\b",
        ),
        _patterns(
            r"\b(?:no|without)\s+central\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b",
            r"\bno\s+(?:embolus|thrombus|filling defect)\b[^.;\n]{0,50}"
            r"\b(?:main pulmonary arter(?:y|ies)|pulmonary trunk)\b",
        ),
    ),
    "lobar": (
        _patterns(
            r"\blobar\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b",
            r"\b(?:embolus|thrombus|filling defect)\b[^.;\n]{0,40}\blobar arter(?:y|ies)\b",
        ),
        _patterns(r"\b(?:no|without)\s+lobar\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b"),
    ),
    "segmental": (
        _patterns(
            r"\bsegmental\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b",
            r"\b(?:embolus|thrombus|filling defect)\b[^.;\n]{0,40}\bsegmental arter(?:y|ies)\b",
        ),
        _patterns(r"\b(?:no|without)\s+segmental\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b"),
    ),
    "subsegmental": (
        _patterns(
            r"\bsubsegmental\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b",
            r"\b(?:embolus|thrombus|filling defect)\b[^.;\n]{0,40}\bsubsegmental arter(?:y|ies)\b",
        ),
        _patterns(r"\b(?:no|without)\s+subsegmental\s+(?:pulmonary\s+)?embol(?:us|i|ism)\b"),
    ),
    "rv_enlargement": (
        _patterns(
            r"\bright vent(?:ricle|ricular)[^.;\n]{0,25}\b(?:enlarged|enlargement|dilat(?:ed|ation))\b",
            r"\b(?:enlarged|dilated)\s+right ventricle\b",
        ),
        _patterns(
            r"\bno\s+right vent(?:ricular|ricle)[^.;\n]{0,20}\b(?:enlargement|dilatation)\b",
            r"\bright ventricle\s+is\s+not\s+(?:enlarged|dilated)\b",
        ),
    ),
    "rv_lv_ratio_mentioned": (
        _patterns(r"\b(?:rv\s*[/:-]\s*lv|right vent(?:ricular|ricle)[- /]+left vent(?:ricular|ricle))\s+ratio\b"),
        (),
    ),
    "rv_lv_ratio_abnormal": (
        _patterns(
            r"\b(?:rv\s*[/:-]\s*lv|right ventricular[- /]+left ventricular)\s+ratio"
            r"[^.;\n]{0,25}\b(?:elevated|abnormal|increased|greater than one|greater than 1)\b"
        ),
        _patterns(
            r"\b(?:rv\s*[/:-]\s*lv|right ventricular[- /]+left ventricular)\s+ratio"
            r"[^.;\n]{0,25}\b(?:normal|not elevated|less than one|less than 1)\b"
        ),
    ),
    "septal_bowing": (
        _patterns(r"\b(?:interventricular\s+)?septal\s+bowing\b"),
        _patterns(r"\b(?:no|without)\s+(?:interventricular\s+)?septal\s+bowing\b"),
    ),
    "contrast_reflux": (
        _patterns(
            r"\b(?:contrast\s+)?reflux(?:es|ed)?\b[^.;\n]{0,45}\b(?:ivc|inferior vena cava|hepatic veins?)\b",
            r"\breflux\s+of\s+contrast\b",
        ),
        _patterns(
            r"\bno\s+(?:contrast\s+)?reflux\b",
            r"\bwithout\s+reflux\s+of\s+contrast\b",
        ),
    ),
    "pleural_effusion": (
        _patterns(r"\bpleural effusions?\b"),
        _patterns(r"\b(?:no|without)\b[^.;\n]{0,25}\bpleural effusions?\b"),
    ),
    "pericardial_effusion": (
        _patterns(r"\bpericardial effusions?\b"),
        _patterns(r"\b(?:no|without)\b[^.;\n]{0,25}\bpericardial effusions?\b"),
    ),
    "malignancy_related_finding": (
        _patterns(
            r"\b(?:metastatic disease|metastases|known malignancy|malignant mass)\b",
            r"\b(?:mass|lesion|nodule)\b[^.;\n]{0,40}\b(?:suspicious for|consistent with)\s+malignan(?:cy|t)\b",
        ),
        _patterns(r"\bno\s+(?:evidence of\s+)?(?:thoracic\s+)?malignan(?:cy|t disease)\b"),
    ),
    "chronic_lung_disease": (
        _patterns(r"\bchronic\s+(?:obstructive\s+)?lung disease\b", r"\bcopd\b"),
        _patterns(r"\bno\s+(?:evidence of\s+)?chronic\s+(?:obstructive\s+)?lung disease\b"),
    ),
    "fibrosis": (
        _patterns(r"\b(?:pulmonary|interstitial) fibrosis\b", r"\bfibrotic\s+(?:lung\s+)?(?:change|disease)s?\b"),
        _patterns(r"\bno\s+(?:evidence of\s+)?(?:pulmonary|interstitial) fibrosis\b"),
    ),
    "emphysema": (
        _patterns(r"\bemphysema(?:tous)?\b"),
        _patterns(r"\bno\s+(?:evidence of\s+)?emphysema(?:tous changes)?\b"),
    ),
}

_RATIO_VALUE = re.compile(
    r"\b(?:rv\s*[/:-]\s*lv|right ventricular[- /]+left ventricular)\s+ratio"
    r"\s*(?:is|=|of|measures?)\s*([0-9]+(?:\.[0-9]+)?)\b",
    _FLAGS,
)
_ACUITY_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    ("acute_on_chronic", re.compile(r"\bacute[- ]on[- ]chronic\s+(?:pulmonary\s+)?embol", _FLAGS)),
    ("acute", re.compile(r"\bacute\s+(?:pulmonary\s+)?embol", _FLAGS)),
    ("chronic", re.compile(r"\bchronic\s+(?:pulmonary\s+)?embol", _FLAGS)),
    ("uncertain", re.compile(r"\b(?:age-indeterminate|indeterminate acuity)\s+(?:pulmonary\s+)?embol", _FLAGS)),
)


def _sentence_for(text: str, start: int, end: int) -> str:
    for match in _SENTENCE.finditer(text):
        if match.start() <= start and end <= match.end():
            return match.group(0)
    return text[max(0, start - 80): min(len(text), end + 80)]


def _span(match: re.Match[str], text: str) -> EvidenceSpan:
    return EvidenceSpan(text=text[match.start():match.end()], start=match.start(), end=match.end())


def _context_uncertain(text: str, match: re.Match[str]) -> bool:
    sentence = _sentence_for(text, match.start(), match.end())
    return bool(_UNCERTAIN.search(sentence) or _HISTORICAL.search(sentence))


def _overlaps(match: re.Match[str], spans: list[tuple[int, int]]) -> bool:
    return any(match.start() < end and start < match.end() for start, end in spans)


def _boolean_decision(
    text: str,
    target: str,
    positive: tuple[Pattern[str], ...],
    negative: tuple[Pattern[str], ...],
) -> RuleDecision:
    negative_matches = [match for pattern in negative for match in pattern.finditer(text)]
    negative_spans = [(match.start(), match.end()) for match in negative_matches]
    positive_matches = [
        match
        for pattern in positive
        for match in pattern.finditer(text)
        if not _overlaps(match, negative_spans)
    ]
    relevant = [*positive_matches, *negative_matches]
    if any(_context_uncertain(text, match) for match in relevant):
        evidence = next((_span(match, text) for match in relevant if _context_uncertain(text, match)), None)
        return RuleDecision(False, None, f"context_uncertain_{target}", evidence)
    if positive_matches and negative_matches:
        return RuleDecision(False, None, f"conflicting_explicit_{target}", _span(relevant[0], text))
    if positive_matches:
        return RuleDecision(True, True, f"explicit_positive_{target}", _span(positive_matches[0], text))
    if negative_matches:
        return RuleDecision(True, False, f"explicit_negative_{target}", _span(negative_matches[0], text))
    return RuleDecision(False, None, f"no_explicit_evidence_{target}")


def _pe_decision(text: str) -> RuleDecision:
    return _boolean_decision(text, "pe_present", _PE_POSITIVE, _PE_NEGATIVE)


def _acuity_decision(text: str) -> RuleDecision:
    matches: list[tuple[str, re.Match[str]]] = []
    for value, pattern in _ACUITY_PATTERNS:
        matches.extend((value, match) for match in pattern.finditer(text))
    if not matches:
        return RuleDecision(False, None, "acuity_unresolved")
    if any(_context_uncertain(text, match) for _, match in matches):
        value, match = next((item for item in matches if _context_uncertain(text, item[1])))
        return RuleDecision(False, None, f"context_uncertain_acuity:{value}", _span(match, text))
    values = {value for value, _ in matches}
    if "acute_on_chronic" in values:
        values.discard("acute")
        values.discard("chronic")
    if len(values) != 1:
        return RuleDecision(False, None, "conflicting_explicit_acuity", _span(matches[0][1], text))
    value = next(iter(values))
    match = next(match for candidate, match in matches if candidate == value)
    return RuleDecision(True, value, f"explicit_{value}", _span(match, text))


def _ratio_value_decision(text: str) -> RuleDecision:
    matches = list(_RATIO_VALUE.finditer(text))
    if not matches:
        return RuleDecision(False, None, "rv_lv_ratio_value_unresolved")
    if any(_context_uncertain(text, match) for match in matches):
        match = next(match for match in matches if _context_uncertain(text, match))
        return RuleDecision(False, None, "context_uncertain_rv_lv_ratio_value", _span(match, text))
    values = {float(match.group(1)) for match in matches}
    if len(values) != 1:
        return RuleDecision(False, None, "conflicting_rv_lv_ratio_values", _span(matches[0], text))
    value = next(iter(values))
    return RuleDecision(True, value, "explicit_numeric_rv_lv_ratio", _span(matches[0], text))


def apply_rule(report: str, target: str) -> RuleDecision:
    canonical = canonical_target(target)
    text = str(report)
    if not text.strip():
        return RuleDecision(False, None, "empty_report")
    if canonical == "pe_present":
        return _pe_decision(text)
    if canonical == "acuity":
        return _acuity_decision(text)
    if canonical == "rv_lv_ratio_value":
        return _ratio_value_decision(text)
    positive, negative = _BOOLEAN_PATTERNS[canonical]
    return _boolean_decision(text, canonical, positive, negative)
