from __future__ import annotations

from collections.abc import Mapping


def route_confidence(response: Mapping[str, object], threshold: float) -> bool:
    confidence = response.get("confidence")
    if not isinstance(confidence, (int, float)):
        return False
    return float(confidence) >= float(threshold)
