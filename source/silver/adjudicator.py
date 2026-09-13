from __future__ import annotations

from typing import Any

from .schema import SilverLabel


def adjudicate(
    identifiers: dict[str, str],
    target: str,
    falcon: dict[str, Any],
    medgemma: dict[str, Any],
    falcon_model_id: str | None = None,
    medgemma_model_id: str | None = None,
    minimum_confidence: float = 0.0,
) -> SilverLabel:
    first = falcon.get("value")
    second = medgemma.get("value")
    confidences = [
        float(payload["confidence"])
        for payload in (falcon, medgemma)
        if isinstance(payload.get("confidence"), (int, float))
    ]
    agreement_confidence = min(confidences) if len(confidences) == 2 else None
    if (
        first is not None
        and first == second
        and agreement_confidence is not None
        and agreement_confidence >= float(minimum_confidence)
    ):
        return SilverLabel(
            **identifiers,
            target=target,
            value=first,
            status="accepted",
            source="falcon_medgemma_agree",
            confidence=agreement_confidence,
            reason="falcon_medgemma_agree",
            falcon_value=first,
            medgemma_value=second,
            provider="falcon_medgemma_agree",
            model_id=f"{falcon_model_id}+{medgemma_model_id}" if falcon_model_id or medgemma_model_id else None,
        )
    return SilverLabel(
        **identifiers,
        target=target,
        value=None,
        status="abstained",
        source="falcon_medgemma_agree",
        reason=(
            "falcon_medgemma_disagree_no_expert_review"
            if first != second
            else "falcon_medgemma_agreement_below_confidence_threshold"
        ),
        falcon_value=first,
        medgemma_value=second,
        provider="falcon_medgemma_agree",
        model_id=f"{falcon_model_id}+{medgemma_model_id}" if falcon_model_id or medgemma_model_id else None,
    )
