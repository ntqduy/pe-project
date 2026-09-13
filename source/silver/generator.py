from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from .adjudicator import adjudicate
from .audit import AuditRecord, evidence_from_outputs, utc_timestamp
from .confidence import route_confidence
from .falcon import FalconExtractor
from .medgemma import MedGemmaExtractor
from .rules import RULE_VERSION, apply_rule
from .schema import TARGETS, SilverLabel

# A method name IS its source list, in cascade order. Reading the name tells you exactly
# which extractors ran and in which order, so no separate SL00/SL01/SL02 legend is needed.
SILVER_METHODS: dict[str, tuple[str, ...]] = {
    "rule": ("rule",),
    "falcon": ("falcon",),
    "medgemma": ("medgemma",),
    "rule_falcon": ("rule", "falcon"),
    "rule_medgemma": ("rule", "medgemma"),
    "falcon_medgemma": ("falcon", "medgemma"),
    "rule_falcon_medgemma": ("rule", "falcon", "medgemma"),
}


class SilverGenerator:
    """Deterministic cascade over the sources named by ``method``.

    One rule holds for every method: the cheapest deterministic source that can settle a
    target wins, and nothing downstream is asked. Concretely, in stage order,

      rule      an explicit regex hit accepts immediately (no model is called)
      one LLM   a value above the confidence threshold accepts; otherwise abstain
      two LLMs  Falcon first; if it is not confident, MedGemma runs and the two are
                adjudicated -- agreement accepts, disagreement abstains

    Abstaining is always preferred to guessing: a silver label is auxiliary supervision,
    and a wrong accepted row is worse than a missing one.
    """

    def __init__(
        self,
        method: str,
        *,
        falcon: FalconExtractor | None = None,
        medgemma: MedGemmaExtractor | None = None,
        confidence_threshold: float = 0.8,
        medgemma_confidence_threshold: float | None = None,
        agreement_confidence_threshold: float | None = None,
        prompt_version: str | None = None,
        run_id: str | None = None,
    ):
        normalized = str(method).strip().lower()
        if normalized not in SILVER_METHODS:
            raise ValueError(
                f"unsupported silver method: {method}; "
                f"expected one of {sorted(SILVER_METHODS)}"
            )
        self.stages = SILVER_METHODS[normalized]
        if "falcon" in self.stages and falcon is None:
            raise ValueError(f"{normalized} requires configured Falcon")
        if "medgemma" in self.stages and medgemma is None:
            raise ValueError(f"{normalized} requires configured MedGemma")
        self.method = normalized
        self.falcon = falcon
        self.medgemma = medgemma
        self.confidence_threshold = float(confidence_threshold)
        self.medgemma_confidence_threshold = float(
            confidence_threshold
            if medgemma_confidence_threshold is None
            else medgemma_confidence_threshold
        )
        self.agreement_confidence_threshold = float(
            confidence_threshold
            if agreement_confidence_threshold is None
            else agreement_confidence_threshold
        )
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("medgemma_confidence_threshold", self.medgemma_confidence_threshold),
            ("agreement_confidence_threshold", self.agreement_confidence_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.prompt_version = prompt_version
        self.run_id = run_id or uuid.uuid4().hex

    def _identifiers(self, report: Mapping[str, Any]) -> dict[str, str]:
        values = {name: str(report.get(name) or "") for name in ("patient_id", "study_id", "report_id")}
        if not all(values.values()):
            raise ValueError("report requires patient_id, study_id, and report_id")
        return values

    def generate_report(self, report: Mapping[str, Any]) -> list[SilverLabel]:
        labels, _ = self.generate_report_with_audit(report)
        return labels

    def generate_report_with_audit(
        self, report: Mapping[str, Any]
    ) -> tuple[list[SilverLabel], list[AuditRecord]]:
        identifiers = self._identifiers(report)
        text = str(report.get("report_text") or "")
        labels: list[SilverLabel] = []
        audits: list[AuditRecord] = []
        if not text.strip():
            for target in TARGETS:
                label = SilverLabel(
                    **identifiers, target=target, value=None, status="no_result", source="pipeline",
                    reason="empty_report",
                )
                labels.append(label)
                audits.append(self._audit(label, {}))
            return labels, audits
        for target in TARGETS:
            try:
                label, raw = self._generate_target(identifiers, text, target)
            except Exception as exc:  # noqa: BLE001 - provider/data failures map to no_result per target
                label = SilverLabel(
                    **identifiers, target=target, value=None, status="no_result", source="pipeline",
                    reason=f"technical_failure:{type(exc).__name__}:{exc}",
                )
                raw = {}
            labels.append(label)
            audits.append(self._audit(label, raw))
        return labels, audits

    def _audit(self, label: SilverLabel, raw: Mapping[str, Any]) -> AuditRecord:
        rule_output = raw.get("rule_output")
        falcon_output = raw.get("falcon_output")
        medgemma_output = raw.get("medgemma_output")
        evidence_text, evidence_start, evidence_end = evidence_from_outputs(falcon_output, medgemma_output)
        return AuditRecord(
            patient_id=label.patient_id,
            study_id=label.study_id,
            report_id=label.report_id,
            target=label.target,
            final_status=label.status,
            final_value=label.value,
            final_source=label.source,
            rule_output=rule_output,
            falcon_output=falcon_output,
            medgemma_output=medgemma_output,
            evidence_text=evidence_text,
            evidence_start=evidence_start,
            evidence_end=evidence_end,
            provider=label.provider,
            model_id=label.model_id,
            model_revision=label.model_revision,
            prompt_version=self.prompt_version,
            run_id=self.run_id,
            timestamp=utc_timestamp(),
        )

    @staticmethod
    def _numeric(payload: Mapping[str, Any]) -> float | None:
        confidence = payload.get("confidence")
        return float(confidence) if isinstance(confidence, (int, float)) else None

    def _rule_stage(
        self, identifiers: dict[str, str], text: str, target: str, raw: dict[str, Any]
    ) -> SilverLabel | None:
        rule = apply_rule(text, target)
        raw["rule_output"] = rule.as_dict()
        if not rule.resolved:
            return None
        return SilverLabel(
            **identifiers,
            target=target,
            value=rule.value,
            status="accepted",
            source="rule",
            reason=rule.reason,
            rule_version=RULE_VERSION,
            provider="rule",
        )

    def _falcon_stage(
        self, identifiers: dict[str, str], text: str, target: str, raw: dict[str, Any]
    ) -> tuple[dict[str, Any], SilverLabel | None]:
        """Run Falcon; return its raw output plus an accepted label when it is confident."""
        falcon = self.falcon.extract(text, target)
        raw["falcon_output"] = dict(falcon)
        if falcon.get("value") is None or not route_confidence(falcon, self.confidence_threshold):
            return falcon, None
        return falcon, SilverLabel(
            **identifiers,
            target=target,
            value=falcon["value"],
            status="accepted",
            source="falcon",
            confidence=float(falcon["confidence"]),
            reason="falcon_confident",
            falcon_value=falcon["value"],
            provider="falcon",
            model_id=self.falcon.provider.model_id,
        )

    def _medgemma_only(
        self, identifiers: dict[str, str], text: str, target: str, raw: dict[str, Any]
    ) -> SilverLabel:
        prediction = self.medgemma.extract(text, target)
        raw["medgemma_output"] = dict(prediction)
        value = prediction.get("value")
        if value is None or not route_confidence(
            prediction, self.medgemma_confidence_threshold
        ):
            return SilverLabel(
                **identifiers,
                target=target,
                value=None,
                status="abstained",
                source="medgemma",
                confidence=self._numeric(prediction),
                reason="medgemma_uncertain_or_below_confidence_threshold",
                medgemma_value=value,
                provider="medgemma",
                model_id=self.medgemma.provider.model_id,
            )
        return SilverLabel(
            **identifiers,
            target=target,
            value=value,
            status="accepted",
            source="medgemma",
            confidence=self._numeric(prediction),
            reason="medgemma_confident",
            medgemma_value=value,
            provider="medgemma",
            model_id=self.medgemma.provider.model_id,
        )

    def _generate_target(
        self, identifiers: dict[str, str], text: str, target: str
    ) -> tuple[SilverLabel, dict[str, Any]]:
        raw: dict[str, Any] = {}
        if "rule" in self.stages:
            resolved = self._rule_stage(identifiers, text, target, raw)
            if resolved is not None:
                return resolved, raw

        models = tuple(stage for stage in self.stages if stage in {"falcon", "medgemma"})
        if not models:
            # rule-only: an unresolved regex is the final answer, and it is an abstention.
            return (
                SilverLabel(
                    **identifiers,
                    target=target,
                    value=None,
                    status="abstained",
                    source="rule",
                    reason=str(raw["rule_output"].get("reason") or "rule_unresolved"),
                    rule_version=RULE_VERSION,
                    provider="rule",
                ),
                raw,
            )
        if models == ("medgemma",):
            return self._medgemma_only(identifiers, text, target, raw), raw

        falcon, accepted = self._falcon_stage(identifiers, text, target, raw)
        if accepted is not None:
            return accepted, raw
        if models == ("falcon",):
            return (
                SilverLabel(
                    **identifiers,
                    target=target,
                    value=None,
                    status="abstained",
                    source="falcon",
                    confidence=self._numeric(falcon),
                    reason="falcon_uncertain",
                    falcon_value=falcon.get("value"),
                    provider="falcon",
                    model_id=self.falcon.provider.model_id,
                ),
                raw,
            )
        medgemma = self.medgemma.extract(text, target)
        raw["medgemma_output"] = dict(medgemma)
        return (
            adjudicate(
                identifiers,
                target,
                falcon,
                medgemma,
                self.falcon.provider.model_id,
                self.medgemma.provider.model_id,
                self.agreement_confidence_threshold,
            ),
            raw,
        )

    def generate(self, reports: list[Mapping[str, Any]]) -> list[SilverLabel]:
        seen: set[str] = set()
        output: list[SilverLabel] = []
        for report in reports:
            report_id = str(report.get("report_id") or "")
            if report_id in seen:
                raise ValueError(f"duplicate report_id: {report_id}")
            seen.add(report_id)
            output.extend(self.generate_report(report))
        return output
