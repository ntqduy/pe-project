from __future__ import annotations

from .providers import StructuredProvider, parse_json_response
from .schema import validate_target_value


class FalconExtractor:
    def __init__(self, provider: StructuredProvider, expected_model_id: str):
        if provider.model_id != expected_model_id:
            raise ValueError(f"Falcon model mismatch: expected {expected_model_id}, got {provider.model_id}")
        self.provider = provider

    def extract(self, report: str, target: str) -> dict[str, object]:
        response = parse_json_response(self.provider.predict_target(report, target), target)
        response["value"] = validate_target_value(target, response.get("value"))
        return response
