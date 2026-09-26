from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

from .schema import canonical_target, normalize_target_value, target_spec, validate_target_value

# Stored in the audit so a failed or surprising answer can be inspected afterwards.
RAW_RESPONSE_KEY = "_raw_response"
MAX_STORED_RAW_CHARS = 2000


class StructuredProvider(Protocol):
    model_id: str

    def predict_target(self, report: str, target: str) -> Mapping[str, Any]: ...


class ProviderResponseError(ValueError):
    """An unusable model answer; ``raw_response`` keeps the text for the audit trail."""

    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ProviderResponseError("model response did not contain a JSON object", raw_response=text)


class TransformersProvider:
    """Local Hugging Face text-generation provider with a strict JSON response contract."""

    def __init__(
        self,
        model_id: str,
        *,
        auto_model_class: str = "AutoModelForCausalLM",
        tokenizer_class: str = "AutoTokenizer",
        device_map: Any = "auto",
        torch_dtype: str = "auto",
        # Room for value, confidence, a one-sentence reason and a verbatim evidence quote;
        # 192 tokens cut some answers mid-JSON.
        max_new_tokens: int = 384,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        revision: str | None = None,
        prompt_template: str | None = None,
    ):
        try:
            import torch
            import transformers
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers and PyTorch are required for local silver models") from exc
        model_path = Path(model_id)
        if local_files_only and not model_path.is_dir():
            raise FileNotFoundError(f"local silver model directory not found: {model_path}")
        try:
            model_loader = getattr(transformers, auto_model_class)
            tokenizer_loader = getattr(transformers, tokenizer_class)
        except AttributeError as exc:
            raise ValueError(f"unsupported Transformers auto class: {exc}") from exc
        load_options: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if revision:
            load_options["revision"] = revision
        if device_map is not None:
            load_options["device_map"] = device_map
        if torch_dtype != "auto":
            if not hasattr(torch, torch_dtype):
                raise ValueError(f"unknown torch dtype: {torch_dtype}")
            load_options["torch_dtype"] = getattr(torch, torch_dtype)
        tokenizer_options = {
            key: value for key, value in load_options.items() if key not in {"device_map", "torch_dtype"}
        }
        self.tokenizer = tokenizer_loader.from_pretrained(model_id, **tokenizer_options)
        self.model = model_loader.from_pretrained(model_id, **load_options)
        if device_map is None:
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.prompt_template = (prompt_template or _DEFAULT_PROMPT).strip()

    def _prompt(self, report: str, target: str, feedback: str | None = None) -> str:
        spec = target_spec(target)
        instruction = (
            f"{self.prompt_template}\n"
            f"Target: {spec.name}\n"
            f"Definition: {spec.description}\n"
            f"Allowed value: {spec.allowed_prompt_values}\n"
            f"{_target_hint(spec.name)}"
            "Report:\n"
            f"{report}"
        )
        if feedback:
            instruction += (
                "\n\nYour previous answer could not be used: "
                f"{feedback}\nReply again with only the JSON object."
            )
        template = getattr(self.tokenizer, "apply_chat_template", None)
        self._templated = False
        if callable(template):
            try:
                rendered = template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError):
                rendered = None
            if rendered is not None:
                self._templated = True
                return rendered
        return instruction

    def predict_target(
        self, report: str, target: str, feedback: str | None = None
    ) -> Mapping[str, Any]:
        import torch

        prompt = self._prompt(report, target, feedback)
        # A rendered chat template already starts with <bos>; letting the tokenizer add its
        # own gives Gemma a double BOS ([2, 2, ...]), which degrades its answers.
        special = {"add_special_tokens": False} if self._templated else {}
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt", **special)
        except TypeError:
            encoded = self.tokenizer(text=prompt, return_tensors="pt", **special)
        device = next(
            (parameter.device for parameter in self.model.parameters() if not parameter.is_meta),
            None,
        )
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        continuation = generated[:, input_length:]
        raw = self.tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]
        return {**_first_json_object(raw), RAW_RESPONSE_KEY: raw}


# prompt_version v2: explicit true/false/null rules, the PE-negative implication for the
# location targets, a JSON example, and verbatim evidence.
_DEFAULT_PROMPT = (
    "You extract one structured field from a radiology report (usually a CT pulmonary "
    "angiogram impression).\n"
    "Rules for value:\n"
    "- true: the report states the finding is present (any size, side or grade counts, "
    "e.g. 'small bilateral pleural effusions' -> true).\n"
    "- false: the report states the finding is absent, or an explicit statement rules it out "
    "(e.g. 'No pulmonary embolism' -> pe_present, central, lobar, segmental, subsegmental and "
    "saddle are all false).\n"
    "- null: the report does not mention it, or only hedges (possible, cannot exclude, may "
    "represent). Never guess an unstated finding.\n"
    "confidence: a number from 0 to 1 for how directly the text supports the value; use null "
    "when value is null.\n"
    "evidence_text: the shortest exact quote copied from the report that supports the value, "
    "or null.\n"
    "Answer with one JSON object on one line and nothing else, using JSON true/false/null "
    "(not strings), for example:\n"
    '{"target": "pleural_effusion", "value": true, "confidence": 0.95, '
    '"reason": "report states small bilateral pleural effusions", '
    '"evidence_text": "SMALL BILATERAL PLEURAL EFFUSIONS"}'
)


def _target_hint(target: str) -> str:
    """Extra instruction for targets whose answers are easy to get wrong."""
    hints = {
        "acuity": "Only answer when a pulmonary embolism is present; otherwise null.\n",
        "rv_lv_ratio_value": "Only a number written in the report, e.g. 1.2; otherwise null.\n",
        "rv_lv_ratio_mentioned": (
            "true only if the text names an RV/LV (right-to-left ventricular) ratio; "
            "false is not implied by silence, use null.\n"
        ),
    }
    return hints.get(target, "")


def build_transformers_provider(model_id: str, **kwargs: Any) -> TransformersProvider:
    return TransformersProvider(model_id, **kwargs)


def _confidence(value: Any) -> float | None:
    """Model-reported confidence in [0, 1]; percentages and numeric strings are accepted."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        try:
            value = float(text)
        except ValueError:
            return None
    if not isinstance(value, Real):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return None
    if 1.0 < numeric <= 100.0:
        numeric /= 100.0
    return numeric if numeric <= 1.0 else None


def _evidence(payload: dict[str, Any], report: str | None) -> None:
    """Locate the quoted evidence in the report and set its offsets.

    Models are unreliable at character offsets, so offsets are always recomputed from the
    quote. A quote that is not in the report marks ``evidence_valid=False`` instead of
    failing the whole answer; the generator then refuses to accept it.
    """
    evidence = payload.get("evidence_text")
    if evidence is not None and not isinstance(evidence, str):
        evidence = str(evidence)
    evidence = (evidence or "").strip().strip("'\"")
    payload["evidence_start"] = None
    payload["evidence_end"] = None
    if not evidence:
        payload["evidence_text"] = None
        payload["evidence_valid"] = None
        return
    payload["evidence_text"] = evidence
    if report is None:
        payload["evidence_valid"] = None
        return
    start = report.find(evidence)
    if start < 0:
        start = report.lower().find(evidence.lower())
    if start < 0:
        compact_report = " ".join(report.lower().split())
        compact_evidence = " ".join(evidence.lower().split())
        payload["evidence_valid"] = compact_evidence in compact_report
        return
    payload["evidence_start"] = start
    payload["evidence_end"] = start + len(evidence)
    payload["evidence_valid"] = True


def parse_json_response(
    raw: str | Mapping[str, Any],
    target: str,
    *,
    report: str | None = None,
) -> dict[str, Any]:
    """Validate one model answer into ``target, value, confidence, reason, evidence_*``.

    Tolerant of harmless deviations (extra keys, a missing reason, "yes"/"small" for a
    boolean, "95%" confidence) and records every coercion in ``normalization``. Only an
    answer without a usable value raises ``ProviderResponseError``.
    """
    payload = _first_json_object(raw) if isinstance(raw, str) else dict(raw)
    raw_response = payload.pop(RAW_RESPONSE_KEY, raw if isinstance(raw, str) else None)
    notes: list[str] = []
    allowed = {"target", "value", "confidence", "reason", "evidence_text", "evidence_start", "evidence_end"}
    extra = sorted(set(payload) - allowed)
    if extra:
        notes.append("ignored_fields:" + ",".join(extra))
        for name in extra:
            payload.pop(name)
    expected = canonical_target(target)
    returned = payload.get("target")
    if returned is not None:
        try:
            canonical_returned = canonical_target(str(returned))
        except ValueError:
            canonical_returned = None
            notes.append(f"unknown_target_name:{returned}")
        if canonical_returned is not None and canonical_returned != expected:
            raise ProviderResponseError(f"model answered a different target: {returned}", raw_response)
    payload["target"] = expected
    if "value" not in payload:
        raise ProviderResponseError("model response has no value field", raw_response)
    value, note = normalize_target_value(expected, payload.get("value"))
    if note:
        notes.append(note)
    try:
        payload["value"] = validate_target_value(expected, value)
    except ValueError as exc:
        raise ProviderResponseError(str(exc), raw_response) from exc
    confidence = payload.get("confidence")
    payload["confidence"] = _confidence(confidence)
    if confidence is not None and payload["confidence"] is None:
        notes.append(f"unusable_confidence:{confidence!r}")
    reason = payload.get("reason")
    payload["reason"] = None if reason is None else str(reason)
    _evidence(payload, report)
    payload["normalization"] = notes or None
    payload["raw_response"] = (
        str(raw_response)[:MAX_STORED_RAW_CHARS] if raw_response is not None else None
    )
    return payload
