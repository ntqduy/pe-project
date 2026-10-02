from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .audit import AuditRecord, evidence_from_outputs, utc_timestamp
from .confidence import route_confidence
from .medgemma import MedGemmaExtractor
from .providers import MAX_STORED_RAW_CHARS, ProviderResponseError
from .rules import RULE_VERSION, RuleDecision, apply_rule
from .schema import TARGETS, SilverLabel, target_spec

# An explicit "no pulmonary embolism" rules out every PE location, so these targets are
# false by implication even when the report does not name them one by one.
PE_LOCATION_TARGETS = ("central", "lobar", "segmental", "subsegmental", "saddle")
# pe_consistency reasons for a PE location answered true while pe_present is accepted
# false; qc.py counts both as conflicts.
LOCATION_TRUE_BUT_PE_FALSE = "pe_consistency:location_true_but_pe_present_false"
LOCATION_MEDGEMMA_TRUE_BUT_PE_FALSE = "pe_consistency:location_medgemma_true_but_pe_present_false"
# An accepted acuity while pe_present is accepted false is a contradiction (a conflict);
# while pe_present is undecided it is only unsupported (missing_field in qc.py).
ACUITY_WITHOUT_PE = "pe_consistency:acuity_without_pe"
ACUITY_WITHOUT_PE_DECISION = "pe_consistency:acuity_without_pe_decision"

# A method name IS its source list. MedGemma is the only generation method this project
# offers; the deterministic rules only back it up (rule_rescue), they are not a stage.
SILVER_METHODS: dict[str, tuple[str, ...]] = {
    "medgemma": ("medgemma",),
}
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
# Reason fragments of rows whose value does not come from a completed MedGemma call; such a
# report must be asked again on resume even when a rule or pe_consistency filled the row.
TECHNICAL_FAILURE_MARKERS = ("technical_failure", "after_medgemma_failure")


def resolve_confidence_threshold(silver_config: Mapping[str, Any]) -> float:
    """The one MedGemma acceptance threshold, for the generator and the recorded CSVs alike.

    ``medgemma_confidence_threshold`` wins; the older generic ``confidence_threshold`` key
    is still read when it is the only one configured.
    """
    value = silver_config.get("medgemma_confidence_threshold")
    if value is None:
        value = silver_config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    return float(value)


def has_technical_failure(
    rows: Sequence[Mapping[str, Any]], audits: Sequence[Mapping[str, Any]] = ()
) -> bool:
    """Whether any target of one report hit a provider/pipeline failure."""
    if any(str(row.get("status")) == "no_result" for row in rows):
        return True
    if any(marker in str(row.get("reason") or "") for row in rows for marker in TECHNICAL_FAILURE_MARKERS):
        return True
    return any(
        isinstance(record.get("medgemma_output"), Mapping) and "error" in record["medgemma_output"]
        for record in audits
    )


class SilverGenerator:
    """MedGemma labels every target; the explicit rules (rules.py) only back it up.

    Per target:

      MedGemma  a value at or above the confidence threshold, with evidence found in the
                report, accepts; otherwise the target abstains
      rule      with rule_rescue, an unambiguous explicit rule answer vetoes a confident
                MedGemma value it contradicts, and fills an abstention or a technical
                failure unless MedGemma answered the opposite value
      report    with pe_consistency, the PE location/acuity targets are made to agree with
                pe_present (see _enforce_pe_consistency)

    Abstaining is always preferred to guessing: a silver label is auxiliary supervision,
    and a wrong accepted row is worse than a missing one.
    """

    def __init__(
        self,
        method: str,
        *,
        medgemma: MedGemmaExtractor | None = None,
        medgemma_confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
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
        if "medgemma" in self.stages and medgemma is None:
            raise ValueError(f"{normalized} requires configured MedGemma")
        self.method = normalized
        self.medgemma = medgemma
        self.medgemma_confidence_threshold = float(medgemma_confidence_threshold)
        if not 0.0 <= self.medgemma_confidence_threshold <= 1.0:
            raise ValueError("medgemma_confidence_threshold must be in [0, 1]")
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

        pe_present accepted false    -> a location MedGemma (or a rule) called true, accepted
                                        or not, abstains as a conflict; every other location
                                        (MedGemma said false or null) is false by implication.
        pe_present not accepted      -> a location "false" is unsupported: without knowing
                                        whether there is a PE, "no lobar PE" is a guess.
        pe_present not accepted true -> an accepted acuity is withdrawn: it would describe
                                        a PE that is not established.
        A location filled over a technical failure says so in its reason, so the report is
        not frozen in the resume cache.
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
                            reason=LOCATION_TRUE_BUT_PE_FALSE)
                    continue
                # A MedGemma "true" that was not accepted (unconfident, or vetoed by a
                # rule) still disagrees with pe_present; overwriting it with false would
                # hide the conflict that a confident "true" reports.
                if current.medgemma_value is True:
                    replace(target, value=None, status="abstained",
                            reason=LOCATION_MEDGEMMA_TRUE_BUT_PE_FALSE)
                    continue
                reason = "pe_consistency:implied_by_pe_present_false"
                if current.status == "no_result":
                    reason += "_after_technical_failure"
                replace(target, value=False, status="accepted", source="rule", provider="rule",
                        model_id=None, confidence=None, rule_version=RULE_VERSION,
                        reason=reason)
            elif pe_value is None and current.status == "accepted" and current.value is False:
                replace(target, value=None, status="abstained",
                        reason="pe_consistency:location_false_without_pe_decision")
        acuity = labels[by_target["acuity"]]
        if pe_value is not True and acuity.status == "accepted":
            replace("acuity", value=None, status="abstained",
                    reason=ACUITY_WITHOUT_PE if pe_value is False else ACUITY_WITHOUT_PE_DECISION)
        return labels

    def _audit(self, label: SilverLabel, raw: Mapping[str, Any]) -> AuditRecord:
        rule_output = raw.get("rule_output")
        medgemma_output = raw.get("medgemma_output")
        evidence_text, evidence_start, evidence_end = evidence_from_outputs(medgemma_output)
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

    def _rule_decision(self, text: str, target: str, raw: dict[str, Any]) -> RuleDecision:
        """Explicit regex decision, or the PE-negative implication for location targets.

        The implication only fills a location the report does not mention at all; a
        hedged or contradictory location mention stays unresolved.
        """
        decision = apply_rule(text, target)
        if decision.reason.startswith("no_explicit_evidence") and target in PE_LOCATION_TARGETS:
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
            # A rule only fills the abstention when it is the sole explicit answer: an
            # unconfident MedGemma value of the opposite sign makes the target ambiguous.
            if value is None or self._same_value(target, value, decision.value):
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
                reason=f"medgemma_rule_conflict_unconfident:rule={decision.value!r}",
                medgemma_value=value,
                provider="medgemma",
                model_id=model_id,
            )
        if evidence_rejected:
            reason = "medgemma_evidence_not_in_report"
        elif value is None:
            reason = "medgemma_null_value"
        else:
            reason = "medgemma_below_confidence_threshold"
        return SilverLabel(
            **identifiers,
            target=target,
            value=None,
            status="abstained",
            source="medgemma",
            confidence=self._numeric(prediction),
            reason=reason,
            medgemma_value=value,
            provider="medgemma",
            model_id=model_id,
        )

    def _generate_target(
        self, identifiers: dict[str, str], text: str, target: str
    ) -> tuple[SilverLabel, dict[str, Any]]:
        raw: dict[str, Any] = {}
        return self._medgemma_only(identifiers, text, target, raw), raw

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
