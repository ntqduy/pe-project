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
) -> SilverLabel:
    first = falcon.get("value")
    second = medgemma.get("value")
    if first is not None and first == second:
        return SilverLabel(
            **identifiers,
            target=target,
            value=first,
            status="accepted",
            source="falcon_medgemma_agree",
            confidence=float(falcon["confidence"]) if isinstance(falcon.get("confidence"), (int, float)) else None,
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
        reason="falcon_medgemma_disagree_no_expert_review",
        falcon_value=first,
        medgemma_value=second,
        provider="falcon_medgemma_agree",
        model_id=f"{falcon_model_id}+{medgemma_model_id}" if falcon_model_id or medgemma_model_id else None,
    )
