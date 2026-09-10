from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from source.data.paths import ProjectPaths


class OutputPersistenceError(RuntimeError):
    pass


class OutputCollisionError(RuntimeError):
    pass


FAMILY_PATHS = {
    "segmentation": "segmentation",
    "roi": "roi",
    "silver": "silver_label",
    "foundation": "pretraining/foundation",
    "dapt": "pretraining/dapt",
    "alignment": "pretraining/alignment",
    "shared_encoder": "shared_encoder",
    "silver_encoder_adaptation": "pretraining/silver",
    "diagnosis": "diagnosis",
    "prognosis": "prognosis",
    "contour": "contour",
    "silver_ablation": "ablation/silver",
    "architecture_ablation": "ablation/architecture",
    "remove_roi": "ablation/remove_roi",
    "roi_students": "ablation/roi_students",
    "transfer": "ablation/transfer",
    "counterfactual": "counterfactual",
    "roi_student": "roi_students",
    "concept_bottleneck": "concept_bottleneck",
    "summary": "summary",
    "segmentation_validation": "segmentation_validation",
}
STANDARD_RUN_DIRECTORIES = ("checkpoints", "logs", "qc", "figures")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        with path.open("rb") as handle:
            if handle.read() != payload:
                raise OutputPersistenceError(f"write verification failed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    atomic_write_bytes(path, data)


def verify_writable_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex.encode("ascii")
    probe = directory / f".pe-write-test-{uuid.uuid4().hex}"
    try:
        atomic_write_bytes(probe, token)
    except Exception as exc:
        raise OutputPersistenceError(f"persistent output write test failed at {directory}: {exc}") from exc
    finally:
        probe.unlink(missing_ok=True)


class OutputManager:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.root = paths.assert_persistent_output()

    def family_root(self, family: str) -> Path:
        if family not in FAMILY_PATHS:
            raise KeyError(f"unknown output family: {family}")
        destination = (self.root / FAMILY_PATHS[family]).resolve()
        self.paths.require_within_output(destination)
        return destination

    @staticmethod
    def _output_parts(output_id: str) -> tuple[str, ...]:
        """Validate a flat or nested output identifier.

        Most historical runs use a single identifier such as ``DX_global_single``.
        Matrix experiments need a readable hierarchy (for example
        ``all_patient/EHR_0_h/1_month_mortality/weight_dapt/image_only``).  Treating
        that hierarchy as a filesystem path is safe only when every component is an
        ordinary directory name: no empty pieces, traversal, absolute paths, or Windows
        separators are accepted.
        """
        text = str(output_id or "").strip()
        if not text or "\\" in text or "\x00" in text:
            raise ValueError(f"invalid output ID: {output_id!r}")
        parts = tuple(text.split("/"))
        if any(not part or part in {".", ".."} for part in parts):
            raise ValueError(f"invalid output ID: {output_id!r}")
        if Path(text).is_absolute():
            raise ValueError(f"invalid output ID: {output_id!r}")
        return parts

    def run_dir(self, family: str, experiment_id: str) -> Path:
        """Return a collision-safe run directory for an experiment/output identifier."""
        parts = self._output_parts(experiment_id)
        destination = self.family_root(family).joinpath(*parts).resolve()
        self.paths.require_within_output(destination)
        return destination

    def inspect_collision(self, family: str, experiment_id: str) -> str | None:
        destination = self.run_dir(family, experiment_id)
        if not destination.exists():
            return None
        result_path = destination / "result.json"
        if result_path.is_file():
            try:
                status = json.loads(result_path.read_text(encoding="utf-8")).get("experiment", {}).get("status")
            except (OSError, json.JSONDecodeError):
                status = "invalid_result"
            return str(status or "unknown")
        return "incomplete"

    def ensure_layout(self, run_dir: Path) -> Path:
        destination = self.paths.require_within_output(run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name in STANDARD_RUN_DIRECTORIES:
            (destination / name).mkdir(exist_ok=True)
        return destination

    def prepare(
        self,
        family: str,
        experiment_id: str,
        *,
        resume: bool = False,
        overwrite: bool = False,
    ) -> Path:
        if resume and overwrite:
            raise ValueError("--resume and --overwrite are mutually exclusive")
        verify_writable_directory(self.root)
        destination = self.run_dir(family, experiment_id)
        state = self.inspect_collision(family, experiment_id)
        if state is not None and not resume and not overwrite:
            raise OutputCollisionError(
                f"output already exists ({state}): {destination}; use --resume or --overwrite explicitly"
            )
        if state == "completed" and resume:
            raise OutputCollisionError(f"completed experiment cannot be resumed: {destination}")
        if overwrite and destination.exists():
            self.paths.require_within_output(destination)
            shutil.rmtree(destination)
        self.ensure_layout(destination)
        verify_writable_directory(destination)
        return destination

    def write_config(self, run_dir: Path, config: Mapping[str, Any]) -> Path:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML is required to write config snapshots") from exc
        payload = yaml.safe_dump(dict(config), sort_keys=False).encode("utf-8")
        destination = run_dir / "resolved_config.yaml"
        compatibility = run_dir / "config.yaml"
        existing_path = destination if destination.is_file() else compatibility
        if existing_path.is_file():
            existing = yaml.safe_load(existing_path.read_text(encoding="utf-8")) or {}
            old_hash = existing.get("config_hash")
            new_hash = config.get("config_hash")
            if old_hash and new_hash and old_hash != new_hash:
                raise OutputCollisionError(
                    f"resolved config does not match existing run: {old_hash} != {new_hash}"
                )
            if old_hash == new_hash:
                payload = yaml.safe_dump(dict(config), sort_keys=False).encode("utf-8")
                if not destination.is_file():
                    atomic_write_bytes(destination, payload)
                if not compatibility.is_file():
                    atomic_write_bytes(compatibility, payload)
                return destination
        atomic_write_bytes(destination, payload)
        atomic_write_bytes(compatibility, payload)
        return destination

    def write_lineage(self, run_dir: Path, lineage: Mapping[str, Any]) -> Path:
        destination = run_dir / "lineage.json"
        atomic_write_json(destination, lineage)
        return destination

    def write_environment(self, run_dir: Path, environment: Mapping[str, Any]) -> Path:
        destination = run_dir / "environment.json"
        atomic_write_json(destination, environment)
        return destination

    def write_metrics(self, run_dir: Path, evaluation: Mapping[str, Any]) -> Path:
        destination = run_dir / "metrics.json"
        atomic_write_json(destination, evaluation)
        return destination

    def write_result(self, run_dir: Path, result: Mapping[str, Any]) -> Path:
        self.ensure_layout(run_dir)
        lineage = result.get("lineage")
        if isinstance(lineage, Mapping):
            self.write_lineage(run_dir, lineage)
        environment = result.get("reproducibility")
        if isinstance(environment, Mapping):
            self.write_environment(run_dir, environment)
        evaluation = result.get("evaluation")
        if isinstance(evaluation, Mapping):
            self.write_metrics(run_dir, evaluation)
        destination = run_dir / "result.json"
        atomic_write_json(destination, result)
        return destination


def prepare_resumable_run(
    manager: OutputManager,
    family: str,
    experiment_id: str,
    config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Prepare a data-generation run, reusing verified partial/completed state by default."""
    destination = manager.run_dir(family, experiment_id)
    existed = destination.exists()
    if overwrite:
        destination = manager.prepare(family, experiment_id, overwrite=True)
        existed = False
    elif not existed:
        destination = manager.prepare(family, experiment_id)
    else:
        verify_writable_directory(destination)
        manager.ensure_layout(destination)
    manager.write_config(destination, config)
    return destination, existed


def compact_result(
    config: Mapping[str, Any],
    *,
    status: str,
    data: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    compute: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    reproducibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    experiment = dict(config.get("experiment") or {})
    experiment.update({"status": status, "seed": int(config.get("seed", 42))})
    lineage_keys = (
        "backbone",
        "initialization",
        "source_experiment",
        "source_checkpoint",
        "source_checkpoint_hash",
        "dataset",
        "cohort",
        "split",
        "fold",
        "fold_aware",
        "held_out_fold",
        "task",
        "supervision_type",
        "silver_source",
        "encoder_freeze_policy",
        "peft_configuration",
        "organ_adapter_configuration",
        "fusion_type",
        "random_seed",
        "code_commit",
        "dapt",
        "alignment",
        "silver_method",
    )
    lineage_source = dict(config.get("lineage") or {})
    lineage_source.setdefault("backbone", (config.get("model") or {}).get("backbone"))
    lineage_source.setdefault("initialization", (config.get("lineage") or {}).get("initialization"))
    lineage_source.setdefault("dapt", (config.get("dapt") or {}).get("method"))
    lineage_source.setdefault("alignment", bool(config.get("alignment")) if config.get("alignment") else None)
    lineage_source.setdefault("silver_method", (config.get("silver") or {}).get("method"))
    lineage_source.setdefault("silver_source", lineage_source.get("silver_method"))
    lineage_source.setdefault("dataset", (config.get("data") or {}).get("dataset"))
    lineage_source.setdefault("cohort", (config.get("data") or {}).get("cohort"))
    lineage_source.setdefault("fold", (config.get("data") or {}).get("fold"))
    lineage_source.setdefault("task", (config.get("task") or {}).get("primary_target"))
    lineage_source.setdefault("encoder_freeze_policy", (config.get("peft") or {}).get("method"))
    lineage_source.setdefault("peft_configuration", dict(config.get("peft") or {}))
    lineage_source.setdefault("organ_adapter_configuration", dict(config.get("organ_adapter") or {}))
    lineage_source.setdefault(
        "fusion_type",
        (config.get("fusion") or {}).get("type", (config.get("task") or {}).get("architecture")),
    )
    lineage_source.setdefault("random_seed", int(config.get("seed", 42)))
    lineage_source.setdefault("code_commit", (reproducibility or {}).get("git_commit"))
    lineage = {key: lineage_source.get(key) for key in lineage_keys if key in lineage_source}
    return {
        "experiment": experiment,
        "lineage": lineage,
        "data": dict(data or {}),
        "model": dict(model or {}),
        "compute": dict(compute or {}),
        "evaluation": dict(evaluation or {}),
        "reproducibility": dict(reproducibility or {}),
        "config_hash": config.get("config_hash"),
        "created_unix": time.time(),
    }
