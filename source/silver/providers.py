from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

from .schema import canonical_target, target_spec, validate_target_value


@dataclass(frozen=True)
class ProviderIdentity:
    provider: str
    model_id: str
    revision: str | None = None
    checksum_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model_id.strip():
            raise ValueError("provider identity requires provider and model_id")
        if self.checksum_sha256 is not None:
            digest = self.checksum_sha256.lower().strip()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("provider checksum_sha256 must contain exactly 64 hexadecimal characters")
            object.__setattr__(self, "checksum_sha256", digest)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredProvider(Protocol):
    model_id: str

    def predict_target(self, report: str, target: str) -> Mapping[str, Any]: ...


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
    raise ValueError("model response did not contain a JSON object")


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
        max_new_tokens: int = 192,
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

    def _prompt(self, report: str, target: str) -> str:
        spec = target_spec(target)
        instruction = (
            f"{self.prompt_template}\n"
            f"Target: {spec.name}\n"
            f"Definition: {spec.description}\n"
            f"Allowed value: {spec.allowed_prompt_values}\n"
            "Report:\n"
            f"{report}"
        )
        template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(template):
            try:
                return template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError):
                pass
        return instruction

    def predict_target(self, report: str, target: str) -> Mapping[str, Any]:
        import torch

        prompt = self._prompt(report, target)
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        except TypeError:
            encoded = self.tokenizer(text=prompt, return_tensors="pt")
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
        return _first_json_object(raw)


_DEFAULT_PROMPT = (
    "Extract only the requested target from the current radiology report. "
    "Return one JSON object with exactly target, value, confidence, reason, evidence_text, "
    "evidence_start, evidence_end. Use null when evidence is absent or not definite. "
    "Never infer an unstated finding."
)


def build_transformers_provider(model_id: str, **kwargs: Any) -> TransformersProvider:
    return TransformersProvider(model_id, **kwargs)


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("provider confidence must be a number in [0, 1] or null")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("provider confidence must be finite and in [0, 1]")
    return numeric


def _evidence(payload: dict[str, Any], report: str | None) -> None:
    evidence = payload.get("evidence_text")
    start = payload.get("evidence_start")
    end = payload.get("evidence_end")
    if evidence is not None and not isinstance(evidence, str):
        raise ValueError("provider evidence_text must be a string or null")
    if not evidence:
        if start is not None or end is not None:
            raise ValueError("evidence offsets require non-empty evidence_text")
        payload["evidence_text"] = None
        payload["evidence_start"] = None
        payload["evidence_end"] = None
        return
    if (start is None) != (end is None):
        raise ValueError("provider must return both evidence offsets or neither")
    if start is None and report is not None:
        start = report.find(evidence)
        if start < 0:
            start = report.lower().find(evidence.lower())
        if start < 0:
            raise ValueError("provider evidence_text is not present in the report")
        end = start + len(evidence)
    if start is not None:
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("provider evidence offsets must be valid integer offsets")
        if report is not None and (end > len(report) or report[start:end].lower() != evidence.lower()):
            raise ValueError("provider evidence offsets do not match evidence_text")
    payload["evidence_start"] = start
    payload["evidence_end"] = end


def parse_json_response(
    raw: str | Mapping[str, Any],
    target: str,
    *,
    report: str | None = None,
) -> dict[str, Any]:
    payload = _first_json_object(raw) if isinstance(raw, str) else dict(raw)
    allowed = {
        "target",
        "value",
        "confidence",
        "reason",
        "evidence_text",
        "evidence_start",
        "evidence_end",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unexpected model response fields: {sorted(unknown)}")
    required = {"target", "value", "confidence", "reason", "evidence_text"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("model response missing fields: " + ", ".join(missing))
    expected = canonical_target(target)
    returned = canonical_target(str(payload["target"]))
    if returned != expected:
        raise ValueError(f"model returned wrong target: {payload.get('target')}")
    payload["target"] = expected
    payload["value"] = validate_target_value(expected, payload.get("value"))
    payload["confidence"] = _confidence(payload.get("confidence"))
    reason = payload.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("provider reason must be a string or null")
    _evidence(payload, report)
    return payload
