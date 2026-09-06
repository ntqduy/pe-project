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


class SilverGenerator:
    def __init__(
        self,
        method: str,
        *,
        falcon: FalconExtractor | None = None,
        medgemma: MedGemmaExtractor | None = None,
        confidence_threshold: float = 0.8,
        prompt_version: str | None = None,
        run_id: str | None = None,
    ):
        normalized = method.upper()
        if normalized not in {"SL00", "SL01", "SL02"}:
            raise ValueError(f"unsupported silver method: {method}")
        if normalized in {"SL01", "SL02"} and falcon is None:
            raise ValueError(f"{normalized} requires configured Falcon")
        if normalized in {"SL00", "SL02"} and medgemma is None:
            raise ValueError(f"{normalized} requires configured MedGemma")
        self.method = normalized
        self.falcon = falcon
        self.medgemma = medgemma
        self.confidence_threshold = float(confidence_threshold)
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

    def _generate_target(
        self, identifiers: dict[str, str], text: str, target: str
    ) -> tuple[SilverLabel, dict[str, Any]]:
        if self.method == "SL00":
            prediction = self.medgemma.extract(text, target)
            raw = {"medgemma_output": dict(prediction)}
            value = prediction.get("value")
            if value is None:
                return (
                    SilverLabel(
                        **identifiers, target=target, value=None, status="abstained", source="medgemma_only",
                        reason="medgemma_uncertain", provider="medgemma",
                        model_id=self.medgemma.provider.model_id,
                    ),
                    raw,
                )
            return (
                SilverLabel(
                    **identifiers,
                    target=target,
                    value=value,
                    status="accepted",
                    source="medgemma_only",
                    confidence=float(prediction["confidence"]) if isinstance(prediction.get("confidence"), (int, float)) else None,
                    provider="medgemma",
                    model_id=self.medgemma.provider.model_id,
                ),
                raw,
            )
        rule = apply_rule(text, target)
        raw: dict[str, Any] = {"rule_output": rule.as_dict()}
        if rule.resolved:
            return (
                SilverLabel(
                    **identifiers,
                    target=target,
                    value=rule.value,
                    status="accepted",
                    source="rule",
                    reason=rule.reason,
                    rule_version=RULE_VERSION,
                    provider="rule",
                ),
                raw,
            )
        falcon = self.falcon.extract(text, target)
        raw["falcon_output"] = dict(falcon)
        if falcon.get("value") is not None and route_confidence(falcon, self.confidence_threshold):
            return (
                SilverLabel(
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
                ),
                raw,
            )
        if self.method == "SL01":
            return (
                SilverLabel(
                    **identifiers,
                    target=target,
                    value=None,
                    status="abstained",
                    source="falcon",
                    confidence=float(falcon["confidence"]) if isinstance(falcon.get("confidence"), (int, float)) else None,
                    reason="falcon_uncertain",
                    falcon_value=falcon.get("value"),
                    provider="falcon",
                    model_id=self.falcon.provider.model_id,
                ),
                raw,
            )
        medgemma = self.medgemma.extract(text, target)
        raw["medgemma_output"] = dict(medgemma)
        return adjudicate(identifiers, target, falcon, medgemma, self.falcon.provider.model_id, self.medgemma.provider.model_id), raw

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
