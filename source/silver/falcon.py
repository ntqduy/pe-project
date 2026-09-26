from __future__ import annotations

from .providers import ProviderResponseError, StructuredProvider, parse_json_response
from .schema import validate_target_value


class FalconExtractor:
    def __init__(self, provider: StructuredProvider, expected_model_id: str, *, retries: int = 1):
        if provider.model_id != expected_model_id:
            raise ValueError(f"Falcon model mismatch: expected {expected_model_id}, got {provider.model_id}")
        if int(retries) < 0:
            raise ValueError("retries must be >= 0")
        self.provider = provider
        # One corrective re-ask recovers most malformed answers (prose instead of JSON, a
        # string instead of true/false) without changing what the model is asked.
        self.retries = int(retries)

    def _ask(self, report: str, target: str, feedback: str | None) -> dict[str, object]:
        if feedback is None:
            raw = self.provider.predict_target(report, target)
        else:
            try:
                raw = self.provider.predict_target(report, target, feedback=feedback)
            except TypeError:  # providers without the feedback argument simply re-ask
                raw = self.provider.predict_target(report, target)
        response = parse_json_response(raw, target, report=report)
        response["value"] = validate_target_value(target, response.get("value"))
        return response

    def extract(self, report: str, target: str) -> dict[str, object]:
        """Parsed answer; raises ProviderResponseError after the last failed attempt."""
        feedback: str | None = None
        failures: list[str] = []
        for attempt in range(self.retries + 1):
            try:
                response = self._ask(report, target, feedback)
            except ProviderResponseError as exc:
                failures.append(str(exc))
                feedback = str(exc)
                last = exc
                continue
            response["attempts"] = attempt + 1
            if failures:
                response["previous_errors"] = failures
            return response
        raise ProviderResponseError(
            f"{last} (after {self.retries + 1} attempts)", raw_response=last.raw_response
        )
