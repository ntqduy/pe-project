from __future__ import annotations

import argparse
import csv
import importlib
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from source.data.paths import ProjectPaths
from source.utils.config import (
    infer_compute_strategy,
    load_config,
    parse_devices,
    validate_config,
)


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--gpus", default=None, help="physical GPU IDs, for example 0,1")
    run_state = parser.add_mutually_exclusive_group()
    run_state.add_argument("--resume", action="store_true")
    run_state.add_argument("--overwrite", action="store_true")
    return parser


def resolve_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, args.overrides)
    if args.gpus is not None:
        devices = parse_devices(args.gpus)
        config["compute"]["devices"] = devices
        config["compute"]["strategy"] = infer_compute_strategy(devices)
        config["compute"]["accelerator"] = "cuda" if devices else "cpu"
    config["resume"] = bool(args.resume)
    config["overwrite"] = bool(args.overwrite)
    return validate_config({key: value for key, value in config.items() if key != "config_hash"})


def resolve_manifest(config: Mapping[str, Any], paths: ProjectPaths) -> Path:
    data = dict(config.get("data") or {})
    manifest = Path(str(data["manifest"]))
    return manifest if manifest.is_absolute() else paths.dataset_root_for(config) / manifest


def select_patient_rows(
    rows: Sequence[Mapping[str, Any]], patient_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Select complete patient records while preserving input order."""
    requested = tuple(
        dict.fromkeys(
            str(value).strip() for value in patient_ids or () if str(value).strip()
        )
    )
    if patient_ids is not None and not requested:
        raise ValueError("--patient-id cannot be empty")
    if not requested:
        return [dict(row) for row in rows]
    available = {str(row.get("patient_id") or "").strip() for row in rows}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError("patient_id not found: " + ", ".join(missing))
    selected = set(requested)
    return [dict(row) for row in rows if str(row.get("patient_id") or "").strip() in selected]


class PrecomputedReportDataset:
    """Wrap a dataset to add a ``report_embedding`` tensor from precomputed manifest columns.

    Shared by the image-report alignment stage and the report-only diagnosis baseline so
    both consume report representations the same way instead of each loading a live text
    encoder.
    """

    def __init__(self, base: Any, columns: Sequence[str]):
        self.base = base
        self.columns = tuple(columns)
        if not self.columns:
            raise ValueError("report_embedding_columns is required")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = self.base[index]
        row = self.base.rows[index]
        item["report_embedding"] = torch.tensor(
            [float(row[column]) for column in self.columns],
            dtype=torch.float32,
        )
        return item


def build_dataset(config: Mapping[str, Any], paths: ProjectPaths, split: str) -> Any:
    from source.data.dataset import CTPADataset, ReportEmbeddingDataset
    from source.data.transforms import Compose, CTWindowNormalize, ResizeVolume
    data = dict(config.get("data") or {})
    supervision = dict(config.get("supervision") or {})
    task = dict(config.get("task") or {})
    stage = str((config.get("experiment") or {}).get("stage") or "")
    silver_training = dict(config.get("silver_training") or {})
    transforms: list[Any] = []
    preprocessing = dict(config.get("preprocessing") or {})
    if stage == "diagnosis" and str(task.get("architecture", "soft_moe")) == "report_only":
        return ReportEmbeddingDataset(
            resolve_manifest(config, paths),
            split,
            embedding_columns=tuple(
                (config.get("alignment") or {}).get("report_embedding_columns") or ()
            ),
            label_columns=tuple(data.get("label_columns") or ()),
        )
    if "window" in preprocessing:
        transforms.append(CTWindowNormalize(*preprocessing["window"]))
    if preprocessing.get("shape"):
        transforms.append(ResizeVolume(tuple(preprocessing["shape"])))
    silver_path = supervision.get("silver_labels")
    if silver_path:
        # Silver tables are written by the silver stage under the output root; a relative
        # value resolves there, exactly as preflight resolves it.
        silver_path = Path(str(silver_path))
        if not silver_path.is_absolute():
            silver_path = paths.output_asset(silver_path)
    silver_targets = (
        tuple((silver_training.get("targets") or {}).keys())
        if stage == "silver_encoder_adaptation"
        else tuple(task.get("silver_targets") or ())
    )
    roi_manifest = data.get("roi_manifest")
    if roi_manifest:
        roi_manifest = Path(str(roi_manifest))
        if not roi_manifest.is_absolute():
            roi_manifest = paths.output_asset(roi_manifest)
    return CTPADataset(
        resolve_manifest(config, paths),
        paths.dataset_root_for(config),
        split,
        transform=Compose(transforms) if transforms else None,
        image_column=str(data.get("file_column", "image_path")),
        label_columns=tuple(data.get("label_columns") or ()),
        ehr_columns=tuple(data.get("ehr_columns") or ()),
        pesi_columns=tuple(data.get("pesi_columns") or ()),
        mask_columns=dict(data.get("mask_columns") or {}),
        roi_manifest=roi_manifest,
        roi_mask_ids=dict(data.get("roi_mask_ids") or {}),
        roi_control_for=dict(data.get("roi_control_for") or {}),
        silver_path=silver_path,
        silver_targets=silver_targets if silver_path else (),
    )


def build_training_lineage(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    code_commit: str | None,
    source_checkpoint: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a fully validated checkpoint-lineage payload for a training tool.

    Derives train/validation/test patient-ID provenance from the resolved manifest
    (when one is configured) so ``checkpoint_lineage_errors`` can detect prognosis/
    diagnosis fold leakage precisely instead of falling back to ``fold_aware`` alone.
    Every training tool previously hand-rolled a partial lineage dict that omitted
    most of ``REQUIRED_LINEAGE`` and would fail validation on first checkpoint save;
    this is the single place that now builds a complete payload via
    :func:`source.engine.checkpoint.build_checkpoint_lineage`.
    """
    from source.data.manifests import ManifestError, patient_ids_for_splits, read_rows
    from source.engine.checkpoint import build_checkpoint_lineage

    data = dict(config.get("data") or {})
    manifest_overrides: dict[str, Any] = {}
    if data.get("manifest"):
        try:
            manifest_path = resolve_manifest(config, paths)
            rows = read_rows(manifest_path)
        except (ManifestError, KeyError):
            rows = []
        if rows:
            patient_column = str(data.get("patient_id_column", "patient_id"))
            split_column = str(data.get("split_column", "split"))
            for split in ("train", "validation", "test"):
                manifest_overrides[f"{split}_patient_ids"] = sorted(
                    patient_ids_for_splits(
                        rows, (split,), patient_column=patient_column, split_column=split_column
                    )
                )
            manifest_overrides["patient_provenance_path"] = str(manifest_path)
    manifest_overrides.update(dict(overrides or {}))
    return build_checkpoint_lineage(
        config,
        epoch=0,
        validation_metric=None,
        code_commit=code_commit,
        source_checkpoint=source_checkpoint,
        overrides=manifest_overrides,
    )


def write_parquet_atomic(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and pyarrow are required to write Parquet") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        pd.DataFrame([dict(row) for row in rows]).to_parquet(temporary, index=False)
        os.replace(temporary, destination)
        pd.read_parquet(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    if not rows:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def import_symbol(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"symbol must be module:attribute: {path}")
    module, attribute = path.split(":", 1)
    return getattr(importlib.import_module(module), attribute)
