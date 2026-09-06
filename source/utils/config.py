from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is ambiguous or unsafe."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REPLACE_KEY = "_replace_"
_DELETE_KEY = "_delete_"
_STAGE_PREFIXES = {
    "segmentation": ("SEG",),
    "roi": ("ROI",),
    "silver": ("SL",),
    "foundation": ("F",),
    "dapt": ("D",),
    "alignment": ("AL",),
    "silver_encoder_adaptation": ("SE",),
    "diagnosis": ("DX",),
    "prognosis": ("PR",),
    "contour": ("CT",),
    "ablation": ("SA", "A", "RM", "RS", "TR"),
    "counterfactual": ("CF",),
    "roi_student": ("RS", "KD"),
    "segmentation_validation": ("SEG",),
}


def _yaml() -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigError("PyYAML is required; install the project dependencies") from exc
    return yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings while allowing scientific configs to replace inherited mappings.

    Normal mappings retain the historical recursive-merge behaviour.  A mapping that
    contains ``_replace_: true`` replaces the inherited mapping entirely; ``_delete_``
    accepts a string or list of keys to remove before applying the remaining overrides.
    Both directives are consumed and never appear in the resolved configuration.

    This is intentionally explicit: an empty mapping still means "no override", which
    preserves backward compatibility, while single-task configs can replace a multitask
    ``targets`` mapping without inheriting stale heads.
    """
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        raise ConfigError("deep_merge requires mapping inputs")
    replace = override.get(_REPLACE_KEY, False)
    if not isinstance(replace, bool):
        raise ConfigError(f"{_REPLACE_KEY} must be true or false")
    result = {} if replace else copy.deepcopy(dict(base))
    raw_delete = override.get(_DELETE_KEY, ())
    if isinstance(raw_delete, str):
        delete = (raw_delete,)
    elif isinstance(raw_delete, (list, tuple)) and all(isinstance(item, str) for item in raw_delete):
        delete = tuple(raw_delete)
    else:
        raise ConfigError(f"{_DELETE_KEY} must be a string or list of strings")
    for key in delete:
        result.pop(key, None)
    for key, value in override.items():
        if key in {_REPLACE_KEY, _DELETE_KEY}:
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_string(value: str, env: Mapping[str, str], strict: bool) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if env.get(name):
            return env[name]
        missing.append(name)
        return match.group(0)

    expanded = _ENV_PATTERN.sub(replace, value)
    if strict and missing:
        raise ConfigError("unset environment variable(s): " + ", ".join(sorted(set(missing))))
    return expanded


def expand_environment(value: Any, env: Mapping[str, str] | None = None, strict: bool = False) -> Any:
    environment = os.environ if env is None else env
    if isinstance(value, str):
        return _expand_string(value, environment, strict)
    if isinstance(value, list):
        return [expand_environment(item, environment, strict) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment(item, environment, strict) for item in value)
    if isinstance(value, Mapping):
        return {key: expand_environment(item, environment, strict) for key, item in value.items()}
    return value


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        payload = _yaml().safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ConfigError(f"invalid YAML configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration must contain a mapping: {path}")
    return payload


def _load_with_bases(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ConfigError(f"cyclic config inheritance: {chain}")
    payload = _read_mapping(resolved)
    bases = payload.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ConfigError(f"_base_ must be a string or list: {resolved}")
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(str(base))
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        merged = deep_merge(merged, _load_with_bases(base_path, (*stack, resolved)))
    return deep_merge(merged, payload)


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ConfigError(f"override must be KEY=VALUE: {raw}")
    key, text = raw.split("=", 1)
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ConfigError(f"empty override key: {raw}")
    return parts, _yaml().safe_load(text)


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for raw in overrides:
        parts, value = _parse_override(raw)
        cursor = result
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ConfigError(f"cannot set nested override beneath non-mapping: {raw}")
            cursor = existing
        cursor[parts[-1]] = value
    return result


def parse_devices(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, int):
        devices = [value]
    elif isinstance(value, str):
        try:
            devices = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as exc:
            raise ConfigError(f"invalid GPU list: {value}") from exc
    elif isinstance(value, (list, tuple)):
        try:
            devices = [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid GPU list: {value}") from exc
    else:
        raise ConfigError(f"invalid GPU list type: {type(value).__name__}")
    if any(device < 0 for device in devices) or len(devices) != len(set(devices)):
        raise ConfigError(f"GPU IDs must be unique non-negative integers: {devices}")
    return devices


def infer_compute_strategy(devices: Iterable[int]) -> str:
    """Infer CPU, single-GPU, or DDP execution from the physical GPU list."""
    count = len(tuple(devices))
    return "cpu" if count == 0 else "single" if count == 1 else "ddp"


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    experiment = result.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigError("experiment mapping is required")
    experiment_id = str(experiment.get("id") or "").strip()
    stage = str(experiment.get("stage") or "").strip().lower()
    if stage not in _STAGE_PREFIXES:
        raise ConfigError(f"unsupported experiment stage: {stage!r}")
    if not experiment_id or not any(experiment_id.startswith(prefix) for prefix in _STAGE_PREFIXES[stage]):
        raise ConfigError(f"experiment id {experiment_id!r} does not match stage {stage!r}")
    mode = str((result.get("data") or {}).get("mode") or "").lower()
    if mode != "full":
        raise ConfigError("data.mode must be full; use CLI patient selection for small runs")
    data = result.setdefault("data", {})
    if not isinstance(data, dict):
        raise ConfigError("data must be a mapping")
    data.setdefault("dataset", "unspecified")
    data.setdefault("cohort", "unspecified")
    data.setdefault("patient_id_column", "patient_id")
    data.setdefault("study_id_column", "study_id")
    data.setdefault("split_column", "split")
    if not all(
        isinstance(data.get(key), str) and str(data.get(key)).strip()
        for key in ("dataset", "cohort", "patient_id_column", "study_id_column", "split_column")
    ):
        raise ConfigError(
            "data.dataset, data.cohort, and patient/study/split column names must be non-empty strings"
        )
    fold = data.get("fold")
    fold_column = data.get("fold_column")
    if fold is not None and (not isinstance(fold_column, str) or not fold_column.strip()):
        raise ConfigError("data.fold requires a non-empty data.fold_column")

    run_scope = result.setdefault("run_scope", {"mode": "cli_required"})
    if not isinstance(run_scope, dict):
        raise ConfigError("run_scope must be a mapping")
    scope_mode = str(run_scope.get("mode") or "cli_required").lower()
    if scope_mode not in {"cli_required", "full", "patient", "limited"}:
        raise ConfigError("run_scope.mode must be cli_required, full, patient, or limited")
    if scope_mode == "full" and run_scope.get("allow_full") is not True:
        raise ConfigError("run_scope.mode=full requires allow_full: true")
    if scope_mode == "patient":
        patient_ids = run_scope.get("patient_ids")
        if not isinstance(patient_ids, list) or not patient_ids or any(
            not str(value).strip() for value in patient_ids
        ):
            raise ConfigError("run_scope.mode=patient requires a non-empty patient_ids list")
        run_scope["patient_ids"] = list(dict.fromkeys(str(value).strip() for value in patient_ids))
    if scope_mode == "limited":
        limits = [run_scope.get("max_cases"), run_scope.get("max_reports")]
        configured = [value for value in limits if value is not None]
        try:
            valid_limit = (
                len(configured) == 1
                and not isinstance(configured[0], bool)
                and int(configured[0]) >= 1
            )
        except (TypeError, ValueError):
            valid_limit = False
        if not valid_limit:
            raise ConfigError(
                "run_scope.mode=limited requires exactly one positive max_cases or max_reports"
            )
    run_scope["mode"] = scope_mode
    compute = result.setdefault("compute", {})
    if not isinstance(compute, dict):
        raise ConfigError("compute must be a mapping")
    devices = parse_devices(compute.get("devices", []))
    strategy = str(compute.get("strategy", "auto")).lower()
    if strategy not in {"auto", "single", "ddp", "cpu"}:
        raise ConfigError("compute.strategy must be auto, single, ddp, or cpu")
    if strategy == "auto":
        strategy = infer_compute_strategy(devices)
    if strategy == "ddp" and len(devices) < 2:
        raise ConfigError("DDP requires at least two configured devices")
    if strategy == "single" and len(devices) > 1:
        raise ConfigError("single strategy accepts at most one device")
    if strategy == "cpu" and devices:
        raise ConfigError("cpu strategy does not accept CUDA devices")
    compute["devices"] = devices
    compute["strategy"] = strategy
    compute["accelerator"] = "cpu" if strategy == "cpu" else "cuda"
    result.setdefault("seed", 42)
    result.setdefault("evaluation", {"bootstrap_samples": 2000, "confidence": 0.95})
    scientific = {key: value for key, value in result.items() if key not in {"resume", "overwrite", "config_hash"}}
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":"), default=str)
    result["config_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def load_config(
    path: str | Path,
    overrides: Iterable[str] = (),
    env: Mapping[str, str] | None = None,
    strict_env: bool = False,
) -> dict[str, Any]:
    payload = _load_with_bases(Path(path))
    payload = apply_overrides(payload, overrides)
    payload = expand_environment(payload, env=env, strict=strict_env)
    return validate_config(payload)


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_yaml().safe_dump(dict(config), sort_keys=False), encoding="utf-8")
