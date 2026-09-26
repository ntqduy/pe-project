from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from .adjudicator import adjudicate
from .audit import AuditRecord, evidence_from_outputs, utc_timestamp
from .confidence import route_confidence
from .falcon import FalconExtractor
from .medgemma import MedGemmaExtractor
from .providers import MAX_STORED_RAW_CHARS, ProviderResponseError
from .rules import RULE_VERSION, RuleDecision, apply_rule
from .schema import TARGETS, SilverLabel, target_spec

# An explicit "no pulmonary embolism" rules out every PE location, so these targets are
# false by implication even when the report does not name them one by one.
PE_LOCATION_TARGETS = ("central", "lobar", "segmental", "subsegmental", "saddle")

# A method name IS its source list, in cascade order. Reading the name tells you exactly
# which extractors ran and in which order, so no separate SL00/SL01/SL02 legend is needed.
# MedGemma is the only generation method this project offers. The cascade machinery below
# is unchanged and still stage-ordered, so adding a stage back is a one-line change here.
SILVER_METHODS: dict[str, tuple[str, ...]] = {
    "medgemma": ("medgemma",),
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
        rule_rescue: bool = False,
        pe_consistency: bool = False,
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
        # MedGemma stays the primary labeller. With rule_rescue the deterministic regex
        # extractor (rules.py) (a) vetoes a confident MedGemma value it explicitly
        # contradicts, and (b) supplies the value when MedGemma abstains or fails but the
        # report states the finding explicitly. Rescued rows carry source="rule".
        self.rule_rescue = bool(rule_rescue)
        # Cross-target check on the PE hierarchy of one report (see _enforce_pe_consistency).
        self.pe_consistency = bool(pe_consistency)

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
        raws: list[dict[str, Any]] = []
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
            raws.append(raw)
        if self.pe_consistency:
            labels = self._enforce_pe_consistency(identifiers, labels)
        audits = [self._audit(label, raw) for label, raw in zip(labels, raws, strict=True)]
        return labels, audits

    def _enforce_pe_consistency(
        self, identifiers: dict[str, str], labels: list[SilverLabel]
    ) -> list[SilverLabel]:
        """Make the PE location/acuity targets agree with pe_present within one report.

        pe_present accepted false  -> every location is false (filled when not accepted);
                                      a location accepted true or an acuity is withdrawn.
        pe_present not accepted    -> a location "false" is unsupported: without knowing
                                      whether there is a PE, "no lobar PE" is a guess.
        pe_present accepted true   -> unchanged.
        """
        by_target = {label.target: index for index, label in enumerate(labels)}
        pe = labels[by_target["pe_present"]]
        pe_value = pe.value if pe.status == "accepted" else None

        def replace(target: str, **changes: Any) -> None:
            current = labels[by_target[target]]
            fields = {**current.as_dict(), **changes}
            labels[by_target[target]] = SilverLabel(**fields)

        for target in PE_LOCATION_TARGETS:
            current = labels[by_target[target]]
            if pe_value is False:
                if current.status == "accepted" and current.value is False:
                    continue
                if current.status == "accepted" and current.value is True:
                    replace(target, value=None, status="abstained",
                            reason="pe_consistency:location_true_but_pe_present_false")
                    continue
                replace(target, value=False, status="accepted", source="rule", provider="rule",
                        model_id=None, confidence=None, rule_version=RULE_VERSION,
                        reason="pe_consistency:implied_by_pe_present_false")
            elif pe_value is None and current.status == "accepted" and current.value is False:
                replace(target, value=None, status="abstained",
                        reason="pe_consistency:location_false_without_pe_decision")
        acuity = labels[by_target["acuity"]]
        if pe_value is False and acuity.status == "accepted":
            replace("acuity", value=None, status="abstained",
                    reason="pe_consistency:acuity_without_pe")
        return labels

    def _audit(self, label: SilverLabel, raw: Mapping[str, Any]) -> AuditRecord:
        rule_output = raw.get("rule_output")
        falcon_output = raw.get("falcon_output")
        medgemma_output = raw.get("medgemma_output")
        evidence_text, evidence_start, evidence_end = evidence_from_outputs(falcon_output, medgemma_output)
        rule_evidence = (rule_output or {}).get("evidence") if label.source == "rule" else None
        if isinstance(rule_evidence, Mapping):
            evidence_text = rule_evidence.get("text")
            evidence_start = rule_evidence.get("start")
            evidence_end = rule_evidence.get("end")
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

    def _rule_decision(self, text: str, target: str, raw: dict[str, Any]) -> RuleDecision:
        """Explicit regex decision, or the PE-negative implication for location targets."""
        decision = apply_rule(text, target)
        if not decision.resolved and target in PE_LOCATION_TARGETS:
            pe = apply_rule(text, "pe_present")
            if pe.resolved and pe.value is False:
                decision = RuleDecision(True, False, "implied_by_explicit_no_pe", pe.evidence)
        raw["rule_output"] = decision.as_dict()
        return decision

    @staticmethod
    def _same_value(target: str, first: Any, second: Any) -> bool:
        if target_spec(target).kind == "continuous" and first is not None and second is not None:
            return abs(float(first) - float(second)) <= 0.05
        return first == second

    def _rule_label(
        self,
        identifiers: dict[str, str],
        target: str,
        decision: RuleDecision,
        *,
        medgemma_value: Any,
        why: str,
    ) -> SilverLabel:
        return SilverLabel(
            **identifiers,
            target=target,
            value=decision.value,
            status="accepted",
            source="rule",
            reason=f"rule_{decision.reason}_{why}",
            rule_version=RULE_VERSION,
            provider="rule",
            medgemma_value=medgemma_value,
        )

    def _medgemma_only(
        self, identifiers: dict[str, str], text: str, target: str, raw: dict[str, Any]
    ) -> SilverLabel:
        model_id = self.medgemma.provider.model_id
        try:
            prediction = self.medgemma.extract(text, target)
        except ProviderResponseError as exc:
            raw["medgemma_output"] = {
                "error": str(exc),
                "raw_response": (exc.raw_response or "")[:MAX_STORED_RAW_CHARS] or None,
            }
            if self.rule_rescue:
                decision = self._rule_decision(text, target, raw)
                if decision.resolved:
                    return self._rule_label(
                        identifiers, target, decision, medgemma_value=None, why="after_medgemma_failure"
                    )
            return SilverLabel(
                **identifiers,
                target=target,
                value=None,
                status="no_result",
                source="medgemma",
                reason=f"technical_failure:ProviderResponseError:{exc}",
                provider="medgemma",
                model_id=model_id,
            )
        raw["medgemma_output"] = dict(prediction)
        value = prediction.get("value")
        evidence_rejected = value is not None and prediction.get("evidence_valid") is False
        confident = (
            value is not None
            and not evidence_rejected
            and route_confidence(prediction, self.medgemma_confidence_threshold)
        )
        decision = self._rule_decision(text, target, raw) if self.rule_rescue else None
        if confident:
            if decision is not None and decision.resolved and not self._same_value(target, value, decision.value):
                return SilverLabel(
                    **identifiers,
                    target=target,
                    value=None,
                    status="abstained",
                    source="medgemma",
                    confidence=self._numeric(prediction),
                    reason=f"medgemma_rule_conflict:rule={decision.value!r}",
                    medgemma_value=value,
                    provider="medgemma",
                    model_id=model_id,
                )
            return SilverLabel(
                **identifiers,
                target=target,
                value=value,
                status="accepted",
                source="medgemma",
                confidence=self._numeric(prediction),
                reason=(
                    "medgemma_confident_rule_agrees"
                    if decision is not None and decision.resolved
                    else "medgemma_confident"
                ),
                medgemma_value=value,
                provider="medgemma",
                model_id=model_id,
            )
        if decision is not None and decision.resolved:
            return self._rule_label(
                identifiers, target, decision, medgemma_value=value, why="after_medgemma_abstained"
            )
        return SilverLabel(
            **identifiers,
            target=target,
            value=None,
            status="abstained",
            source="medgemma",
            confidence=self._numeric(prediction),
            reason=(
                "medgemma_evidence_not_in_report"
                if evidence_rejected
                else "medgemma_uncertain_or_below_confidence_threshold"
            ),
            medgemma_value=value,
            provider="medgemma",
            model_id=model_id,
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
