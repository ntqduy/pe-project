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
    "dataset": ("DS",),
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
        if isinstance(value, Mapping):
            # Merge into {} when nothing is inherited, so `_replace_` / `_delete_` are
            # consumed at every depth instead of surviving into the resolved config as a
            # phantom key (e.g. a mask region literally named `_replace_`).
            inherited = result.get(key)
            result[key] = deep_merge(inherited if isinstance(inherited, Mapping) else {}, value)
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


DEFAULT_DATASET_PROFILE = "full_inspect"

# Encoder initialization sources. These are the checkpoint kinds a downstream task may be
# initialized from; the task model itself is identical across all of them, which is the
# whole point -- the comparison is between representations, not between architectures.
ENCODER_INIT_SOURCES = (
    "pretrained",
    "dapt",
    "c0",
    "rspect_multitask",
    "rspect_single",
    "silver",
    "custom",
)
_ENCODER_INIT_ALIASES = {
    "public": "pretrained",
    "original": "pretrained",
    "alignment": "c0",
    "c_0": "c0",
    "c_silver": "silver",
    "silver_encoder": "silver",
    "silver_adaptation": "silver",
    "rspect_multi_task": "rspect_multitask",
    "rspect_single_task": "rspect_single",
}
_ENCODER_INITIALIZATION = {
    "pretrained": "public",
    "dapt": "C_SSL",
    "c0": "C0",
    "rspect_multitask": "RSPECT_multitask",
    "rspect_single": "RSPECT_single",
    "silver": "C_silver",
    "custom": "custom",
}


def active_dataset_profiles() -> tuple[str, ...]:
    """The dataset profiles a run may point at. Imported lazily to avoid a config cycle."""
    try:
        from source.dataset import ACTIVE_PROFILES
    except Exception:  # noqa: BLE001 - config must stay usable without the dataset package
        return ("smoke_30", "test_500_sample", "full_inspect")
    return tuple(ACTIVE_PROFILES)


def resolve_backbone(config: dict[str, Any]) -> dict[str, Any]:
    """Fill ``model`` from the selected entry of the inherited backbone registry.

    ``configs/components/backbone/registry.yaml`` holds one contract per backbone; the
    per-backbone component files only select one by name. That makes the backbone a
    config variable -- ``--set model.backbone=ct_clip`` works from any run config,
    because the registry travels with it -- while each contract still lives in exactly
    one place. Explicit ``model`` keys always win, so a targeted override is still possible.

    The registry itself is removed from the resolved config: it is inherited identically
    everywhere, and leaving it in would make one backbone's contract change the config
    hash of every experiment.
    """
    registry = config.pop("backbones", None)
    model = config.get("model")
    if not isinstance(model, dict):
        return config
    name = str(model.get("backbone") or "").strip().lower().replace("-", "_")
    if not name:
        return config
    model["backbone"] = name
    if not isinstance(registry, Mapping):
        return config
    entry = registry.get(name)
    if entry is None:
        raise ConfigError(
            f"unknown backbone {name!r}; the registry defines {sorted(registry)}"
        )
    if not isinstance(entry, Mapping):
        raise ConfigError(f"backbone registry entry for {name!r} must be a mapping")
    for key, value in entry.items():
        if key == "display_name":
            continue
        model.setdefault(key, value)
    display = str(entry.get("display_name") or name)
    lineage = config.setdefault("lineage", {})
    if isinstance(lineage, dict):
        # The selected backbone is authoritative: a swapped backbone must never inherit
        # the previous backbone's name into its checkpoint lineage.
        lineage["backbone"] = display
    return config


def resolve_encoder_initialization(config: dict[str, Any]) -> dict[str, Any]:
    """Turn ``encoder.init_source`` into the concrete checkpoint the stage will load.

    Diagnosis, prognosis, the probes and the ROI students all consume an image encoder.
    Which weights that encoder starts from is an experiment variable, not a code path:
    one task model, a small set of named upstream stages, or an explicitly supplied
    custom checkpoint.

        encoder:
          backbone: ct_fm          # selects the registry entry (same key as model.backbone)
          init_source: pretrained | dapt | c0 | rspect_multitask | rspect_single |
                       silver | custom
          checkpoint: <explicit path, or null to use encoder.sources[init_source]>

    ``pretrained`` loads the public weights through the backbone contract and transfers
    nothing; the other sources transfer ``lineage.transfer_modules`` out of a project
    checkpoint. Either way the model built afterwards is the same model. ``alignment``
    and ``silver_encoder`` remain user-facing aliases for the legacy ``c0`` and
    ``silver`` source keys.
    """
    encoder = config.get("encoder")
    if not isinstance(encoder, dict):
        return config
    raw = str(encoder.get("init_source") or "").strip().lower().replace("-", "_")
    if not raw:
        return config
    source = _ENCODER_INIT_ALIASES.get(raw, raw)
    if source not in ENCODER_INIT_SOURCES:
        raise ConfigError(
            f"encoder.init_source must be one of {ENCODER_INIT_SOURCES}; got {raw!r}"
        )
    encoder["init_source"] = source
    lineage_block = config.setdefault("lineage", {})
    if isinstance(lineage_block, dict):
        # Recorded in every checkpoint and every result, so a number always says which
        # kind of weights the encoder started from.
        lineage_block["encoder_init_source"] = source
        # Keep the public matrix name (for example ``alignment``) if a launcher supplied
        # it, instead of losing it to the historical internal key (``c0``).
        lineage_block["weight_source"] = str(encoder.get("weight_source") or source)

    backbone = str(encoder.get("backbone") or "").strip().lower().replace("-", "_")
    if backbone:
        model = config.setdefault("model", {})
        if isinstance(model, dict):
            model["backbone"] = backbone

    sources = dict(encoder.get("sources") or {})
    unknown = sorted(set(sources) - set(ENCODER_INIT_SOURCES))
    if unknown:
        raise ConfigError("unknown encoder.sources key(s): " + ", ".join(unknown))
    explicit = encoder.get("checkpoint") not in (None, "")
    checkpoint = encoder.get("checkpoint") or sources.get(source)
    checkpoint = str(checkpoint).strip() if checkpoint not in (None, "") else None
    if checkpoint and not explicit and source != "pretrained":
        # encoder.sources holds the canonical run of each adaptation stage, and that run
        # belongs to one backbone. Silently handing a CT-FM DAPT checkpoint to a TotalFM
        # experiment would produce a number nobody could interpret.
        baseline = str((config.get("experiment") or {}).get("baseline_backbone") or "").strip()
        selected = str((config.get("model") or {}).get("backbone") or "").strip()
        if baseline and selected and selected != baseline:
            raise ConfigError(
                f"encoder.init_source={source} would load the default {baseline} checkpoint "
                f"into a {selected} run. Point encoder.checkpoint at the {selected} "
                f"{source} checkpoint (and set encoder.source_experiment to match)."
            )

    lineage = config.setdefault("lineage", {})
    if not isinstance(lineage, dict):
        raise ConfigError("lineage must be a mapping")
    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        raise ConfigError("model must be a mapping")

    if source == "pretrained":
        if checkpoint:
            raise ConfigError(
                "encoder.init_source=pretrained loads the public backbone weights; "
                "it must not also name an adapted checkpoint"
            )
        lineage["source_checkpoint"] = None
        lineage["source_experiment"] = None
        lineage["initialization"] = _ENCODER_INITIALIZATION[source]
        lineage["dapt"] = "none"
        lineage["alignment"] = False
        model["load_pretrained"] = True
    else:
        if not checkpoint:
            raise ConfigError(
                f"encoder.init_source={source} needs a checkpoint: set encoder.checkpoint "
                f"or encoder.sources.{source}"
            )
        lineage["source_checkpoint"] = checkpoint
        lineage["initialization"] = _ENCODER_INITIALIZATION[source]
        experiments = dict(encoder.get("source_experiments") or {})
        lineage["source_experiment"] = encoder.get("source_experiment") or experiments.get(source)
        lineage.setdefault("transfer_modules", ["image_encoder"])
        lineage["alignment"] = source in {"c0", "silver"}
        if encoder.get("dapt_method"):
            lineage["dapt"] = encoder["dapt_method"]
        if source == "silver":
            # Distinct from lineage.silver_source, which names the silver labels a task is
            # *supervised* with. This one names the silver source the encoder was adapted on.
            lineage["encoder_silver_source"] = encoder.get("silver_source")
        model.setdefault("load_pretrained", False)
        stage = str((config.get("experiment") or {}).get("stage") or "")
        if stage == "silver_encoder_adaptation":
            # This stage reads its starting encoder from init.checkpoint, not from lineage.
            initialization = config.setdefault("init", {})
            if isinstance(initialization, dict):
                initialization["checkpoint"] = checkpoint
                # Do not leave the legacy C0 values inherited from
                # components/task/silver_adaptation.yaml in place. They are a real
                # provenance contract checked by train_silver_encoder.py, so an RSPECT
                # checkpoint must be identified as RSPECT rather than silently relabelled
                # as C0 (and vice versa).
                initialization["encoder"] = _ENCODER_INITIALIZATION[source]
                initialization["experiment"] = lineage.get("source_experiment")
    return config


def stamp_experiment_variant(config: dict[str, Any]) -> dict[str, Any]:
    """Give every (dataset, backbone, encoder-initialization) combination its own run id.

    Two runs that differ only in which weights the encoder started from are different
    experiments and must not overwrite each other's output directory. Each backbone and
    encoder component records its own baseline; only a deviation from that baseline is
    stamped, so a config left at its defaults keeps exactly the id it has today.

        DX_anatomy_concat                     baseline
        DX_anatomy_concat__enc_dapt           same model, DAPT initialization
        DX_anatomy_concat__ds_test_500_sample__bb_ct_clip__enc_silver
    """
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("variant_stamp") is False:
        return config
    identifier = str(experiment.get("id") or "").strip()
    if not identifier:
        return config
    parts: list[str] = []

    profile = str((config.get("data") or {}).get("profile") or "").strip()
    if profile and profile != str(experiment.get("baseline_dataset") or DEFAULT_DATASET_PROFILE):
        parts.append(f"ds_{profile}")

    baseline_backbone = str(experiment.get("baseline_backbone") or "").strip()
    backbone = str((config.get("model") or {}).get("backbone") or "").strip()
    if baseline_backbone and backbone and backbone != baseline_backbone:
        parts.append(f"bb_{backbone}")

    baseline_init = str(experiment.get("baseline_init") or "").strip()
    init_source = str((config.get("encoder") or {}).get("init_source") or "").strip()
    if baseline_init and init_source and init_source != baseline_init:
        parts.append(f"enc_{init_source}")

    for part in parts:
        suffix = f"__{part}"
        if suffix not in identifier:
            identifier += suffix
    experiment["id"] = identifier
    return config


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    experiment = result.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigError("experiment mapping is required")
    stage = str(experiment.get("stage") or "").strip().lower()
    if stage not in _STAGE_PREFIXES:
        raise ConfigError(f"unsupported experiment stage: {stage!r}")
    mode = str((result.get("data") or {}).get("mode") or "").lower()
    if mode != "full":
        raise ConfigError("data.mode must be full; use CLI patient selection for small runs")
    data = result.setdefault("data", {})
    if not isinstance(data, dict):
        raise ConfigError("data must be a mapping")
    # Which cohort this run reads. data.mode stays "full" -- it means "full-data code
    # path", never a pilot shortcut; the cohort size is the dataset profile's business.
    data.setdefault("profile", DEFAULT_DATASET_PROFILE)
    profile = str(data.get("profile") or "").strip()
    if profile not in active_dataset_profiles():
        raise ConfigError(
            f"data.profile must be one of {active_dataset_profiles()}; got {profile!r}"
        )
    data["profile"] = profile
    result = resolve_backbone(result)
    result = resolve_encoder_initialization(result)
    result = stamp_experiment_variant(result)
    experiment = result["experiment"]
    data = result["data"]
    experiment_id = str(experiment.get("id") or "").strip()
    if not experiment_id or not any(experiment_id.startswith(prefix) for prefix in _STAGE_PREFIXES[stage]):
        raise ConfigError(f"experiment id {experiment_id!r} does not match stage {stage!r}")
    # The cohort a checkpoint was trained/adapted on is part of its lineage, so it
    # defaults to the dataset profile rather than to a placeholder.
    data.setdefault("dataset", profile)
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

    # EHR window selection is a prognostic input contract, not just an implementation
    # detail. Persist it in every resulting checkpoint lineage alongside cohort and
    # dataset profile so EHR_0_h and EHR_24_h results remain auditable after export.
    ehr_profile = data.get("ehr_profile")
    if ehr_profile is not None and str(ehr_profile).strip():
        lineage = result.setdefault("lineage", {})
        if not isinstance(lineage, dict):
            raise ConfigError("lineage must be a mapping")
        lineage["ehr_profile"] = str(ehr_profile).strip()

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
