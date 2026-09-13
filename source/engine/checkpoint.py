from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class CheckpointError(RuntimeError):
    pass


CHECKPOINT_SCHEMA_VERSION = 2
REQUIRED_LINEAGE = (
    "experiment_id",
    "stage",
    "backbone",
    "initialization",
    "source_checkpoint",
    "source_checkpoint_hash",
    "dataset",
    "split",
    "fold",
    "task",
    "supervision_type",
    "silver_source",
    "encoder_freeze_policy",
    "peft_configuration",
    "organ_adapter_configuration",
    "fusion_type",
    "random_seed",
    "code_commit",
    "epoch",
    "validation_metric",
)


def unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def strip_ddp_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def checkpoint_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint not found for hashing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonicalize_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(lineage)
    if "code_commit" not in result and "git_commit" in result:
        result["code_commit"] = result.get("git_commit")
    if "git_commit" not in result and "code_commit" in result:
        result["git_commit"] = result.get("code_commit")
    if "silver_source" not in result and "silver_method" in result:
        result["silver_source"] = result.get("silver_method")
    source = result.get("source_checkpoint")
    if source and not result.get("source_checkpoint_hash"):
        candidate = Path(str(source))
        if candidate.is_file():
            result["source_checkpoint_hash"] = checkpoint_sha256(candidate)
    return result


def _validate_lineage(lineage: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_LINEAGE if key not in lineage]
    if missing:
        raise CheckpointError("checkpoint lineage missing: " + ", ".join(missing))
    for key in ("experiment_id", "stage", "backbone", "initialization", "dataset", "task", "supervision_type"):
        if not str(lineage.get(key) or "").strip() or str(lineage.get(key)).lower() == "unspecified":
            raise CheckpointError(f"checkpoint lineage has no explicit {key}")
    source = lineage.get("source_checkpoint")
    source_hash = lineage.get("source_checkpoint_hash")
    if source:
        if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in source_hash
        ):
            raise CheckpointError("source_checkpoint_hash must be a SHA-256 hex digest")
    elif source_hash not in {None, ""}:
        raise CheckpointError("source_checkpoint_hash requires source_checkpoint")
    if not isinstance(lineage.get("peft_configuration"), MappingABC):
        raise CheckpointError("peft_configuration must be a mapping")
    if not isinstance(lineage.get("organ_adapter_configuration"), MappingABC):
        raise CheckpointError("organ_adapter_configuration must be a mapping")
    seed = lineage.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CheckpointError("random_seed must be an integer")


def build_checkpoint_lineage(
    config: Mapping[str, Any],
    *,
    epoch: int,
    validation_metric: float | None,
    code_commit: str | None,
    source_checkpoint: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical, publication-facing checkpoint provenance payload."""
    experiment = dict(config.get("experiment") or {})
    task = dict(config.get("task") or {})
    lineage = dict(config.get("lineage") or {})
    data = dict(config.get("data") or {})
    peft = dict(config.get("peft") or {})
    organ_adapter = dict(config.get("organ_adapter") or {})
    fusion = dict(config.get("fusion") or {})
    effective_stage = str(experiment.get("stage") or "")
    if effective_stage == "ablation":
        effective_stage = str(task.get("base_stage") or effective_stage)
    source = source_checkpoint if source_checkpoint is not None else lineage.get("source_checkpoint")
    configured_hash = lineage.get("source_checkpoint_hash")
    source_hash = configured_hash
    if source and not source_hash:
        candidate = Path(str(source))
        if candidate.is_file():
            source_hash = checkpoint_sha256(candidate)
    payload: dict[str, Any] = {
        "experiment_id": str(experiment.get("id") or ""),
        "stage": str(experiment.get("stage") or ""),
        "backbone": str(lineage.get("backbone") or (config.get("model") or {}).get("backbone") or ""),
        "initialization": str(lineage.get("initialization") or ""),
        "source_experiment": lineage.get("source_experiment"),
        "source_checkpoint": str(source) if source else None,
        "source_checkpoint_hash": source_hash,
        "dataset": str(data.get("dataset") or ""),
        "cohort": str(data.get("cohort") or ""),
        "split": lineage.get("split", "train"),
        "fold": data.get("fold", lineage.get("fold")),
        "fold_aware": bool(lineage.get("fold_aware", False)),
        "held_out_fold": lineage.get("held_out_fold"),
        "task": str(lineage.get("task") or task.get("primary_target") or effective_stage),
        "supervision_type": str(lineage.get("supervision_type") or ""),
        "silver_source": lineage.get("silver_source", lineage.get("silver_method")),
        "encoder_freeze_policy": str(
            lineage.get("encoder_freeze_policy") or peft.get("method") or "full"
        ),
        "peft_configuration": peft,
        "organ_adapter_configuration": organ_adapter,
        "fusion_type": fusion.get("type", task.get("architecture")),
        "random_seed": int(config.get("seed", 42)),
        "code_commit": code_commit,
        "git_commit": code_commit,
        "epoch": int(epoch),
        "validation_metric": validation_metric,
        "dapt": lineage.get("dapt", (config.get("dapt") or {}).get("method")),
        "alignment": lineage.get("alignment"),
        "label_schema": list(data.get("label_columns") or task.get("targets") or ()),
        "architecture": task.get("architecture"),
        "finetuning_strategy": peft.get("method", "full"),
        "config_path": lineage.get("config_path"),
    }
    for optional in (
        "encoder_init_source",
        "weight_source",
        "ehr_profile",
        "encoder_silver_source",
        "train_patient_ids",
        "validation_patient_ids",
        "test_patient_ids",
        "patient_provenance_path",
        "manifest_hash",
        "split_hash",
    ):
        if optional in lineage:
            payload[optional] = lineage[optional]
    payload.update(dict(overrides or {}))
    payload = _canonicalize_lineage(payload)
    _validate_lineage(payload)
    return payload


def save_checkpoint_atomic(
    path: str | Path,
    model: Any,
    *,
    lineage: Mapping[str, Any],
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise CheckpointError("PyTorch is required for checkpoints") from exc
    canonical_lineage = _canonicalize_lineage(lineage)
    _validate_lineage(canonical_lineage)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    metadata_temporary: Path | None = None
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "lineage": canonical_lineage,
        "model_state": strip_ddp_prefix(unwrap_model(model).state_dict()),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "extra": dict(extra or {}),
    }
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        verified = torch.load(destination, map_location="cpu", weights_only=False)
        if verified.get("lineage") != payload["lineage"] or "model_state" not in verified:
            raise CheckpointError(f"checkpoint verification failed: {destination}")
        metadata = {
            "checkpoint_id": checkpoint_sha256(destination),
            "checkpoint_path": str(destination.resolve()),
            "stage": canonical_lineage.get("stage"),
            "source_checkpoint": canonical_lineage.get("source_checkpoint"),
            "dataset": canonical_lineage.get("dataset"),
            "task": canonical_lineage.get("task"),
            "label_schema": canonical_lineage.get("label_schema"),
            "architecture": canonical_lineage.get("architecture"),
            "finetuning_strategy": canonical_lineage.get("finetuning_strategy"),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": canonical_lineage.get("git_commit"),
            "config_path": canonical_lineage.get("config_path"),
        }
        metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
        metadata_temporary = metadata_path.with_name(
            f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
        )
        with metadata_temporary.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(metadata_temporary, metadata_path)
        parsed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if parsed_metadata.get("checkpoint_id") != checkpoint_sha256(destination):
            raise CheckpointError(f"checkpoint metadata verification failed: {metadata_path}")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        if metadata_temporary is not None:
            metadata_temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    model: Any | None = None,
    *,
    modules: tuple[str, ...] | None = None,
    strict: bool = True,
    strict_modules: bool = True,
    map_location: str = "cpu",
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise CheckpointError("PyTorch is required for checkpoints") from exc
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload or "lineage" not in payload:
        raise CheckpointError(f"invalid project checkpoint: {source}")
    payload["lineage"] = _canonicalize_lineage(payload["lineage"])
    _validate_lineage(payload["lineage"])
    state = strip_ddp_prefix(payload["model_state"])
    selected_report: dict[str, Any] | None = None
    if modules:
        prefixes = tuple(prefix.rstrip(".") + "." for prefix in modules)
        state = {key: value for key, value in state.items() if key.startswith(prefixes)}
        if not state:
            raise CheckpointError(f"none of the requested modules were found: {modules}")
        if model is not None:
            target_state = strip_ddp_prefix(unwrap_model(model).state_dict())
            expected = {key for key in target_state if key.startswith(prefixes)}
            actual = set(state)
            missing_selected = sorted(expected - actual)
            unexpected_selected = sorted(actual - expected)
            selected_report = {
                "missing_selected_keys": missing_selected,
                "unexpected_selected_keys": unexpected_selected,
                "selected_source_keys": len(actual),
                "selected_target_keys": len(expected),
            }
            if strict_modules and (missing_selected or unexpected_selected):
                raise CheckpointError(
                    "strict selected-module transfer failed: "
                    f"missing={missing_selected} unexpected={unexpected_selected}"
                )
    if model is not None:
        target = unwrap_model(model)
        target_state = strip_ddp_prefix(target.state_dict())
        shape_mismatches = [
            {
                "key": key,
                "checkpoint_shape": list(value.shape),
                "model_shape": list(target_state[key].shape),
            }
            for key, value in state.items()
            if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
        ]
        if shape_mismatches and (strict or (modules and strict_modules)):
            raise CheckpointError(f"checkpoint shape mismatch: {shape_mismatches}")
        mismatched_keys = {item["key"] for item in shape_mismatches}
        compatible_state = {key: value for key, value in state.items() if key not in mismatched_keys}
        incompatible = target.load_state_dict(compatible_state, strict=strict and not modules)
        payload["load_report"] = {
            "checkpoint_path": str(source.resolve()),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "shape_mismatches": shape_mismatches,
            "loaded_keys": len(compatible_state),
            "encoder_layers_loaded": sum(
                key.startswith("image_encoder.") for key in compatible_state
            ),
            "head_reset": bool(
                modules
                and "image_encoder" in modules
                and any(key.startswith(("head.", "heads.")) for key in target_state)
            ),
            "selected_modules": list(modules or ()),
        }
        if selected_report:
            payload["load_report"].update(selected_report)
    return payload


def inspect_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Read and validate project checkpoint metadata for preflight without building a model."""
    payload = load_checkpoint(path, model=None, strict=False, map_location=map_location)
    return {
        "path": str(Path(path).resolve()),
        "sha256": checkpoint_sha256(path),
        "schema_version": payload.get("schema_version"),
        "lineage": dict(payload["lineage"]),
        "state_key_count": len(payload["model_state"]),
    }
