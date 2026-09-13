from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from source.components.roi.masks import remove_roi
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.gather import gather_prediction_rows
from source.distributed.setup import initialize_distributed, rank_zero_call, wrap_ddp
from source.engine.checkpoint import checkpoint_sha256, load_checkpoint
from source.engine.experiment import OutputManager, compact_result
from source.engine.factory import build_task_model
from source.engine.trainer import move_to_device
from source.metrics.bootstrap import bootstrap_binary_predictions
from source.metrics.classification import binary_classification_metrics, select_threshold_on_validation
from source.roi.counterfactual import MaskingPolicy
from source.utils.config import validate_config
from source.utils.environment import environment_report
from source.utils.seed import seed_everything
from tools._common import base_parser, build_dataset, resolve_cli_config, write_parquet_atomic


def _policy(task: dict[str, Any]) -> MaskingPolicy:
    raw = task.get("masking_policy", "local_mean")
    options = {"type": raw} if isinstance(raw, str) else dict(raw)
    return MaskingPolicy(
        name=str(options.get("type", options.get("name", "local_mean"))),
        local_radius_voxels=int(options.get("local_radius_voxels", 3)),
        noise_scale=float(options.get("noise_scale", 1.0)),
    )


def _loader(config, paths, split, context, *, patient_ids=None, maximum=None):
    dataset = build_dataset(config, paths, split)
    if patient_ids:
        requested = {str(value) for value in patient_ids}
        available = {str(row["patient_id"]) for row in dataset.rows}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"patient_id not found in {split}: {', '.join(missing)}")
        dataset.rows = [row for row in dataset.rows if str(row["patient_id"]) in requested]
    if maximum is not None:
        dataset.rows = dataset.rows[: int(maximum)]
    if not dataset.rows:
        raise ValueError(f"counterfactual selection is empty for split={split}")
    sampler = DistributedSampler(dataset, shuffle=False) if context.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int((config.get("evaluation") or {}).get("batch_size", 1)),
        sampler=sampler,
        shuffle=False,
        num_workers=int((config.get("compute") or {}).get("num_workers", 0)),
    )
    expected = {(str(row["patient_id"]), str(row["study_id"])) for row in dataset.rows}
    return loader, expected


@torch.no_grad()
def _original_prediction_rows(model, loader, primary, label_index, context):
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        moved = move_to_device(batch, context.device)
        logits = model(moved["volume"], moved["masks"])["logits"][primary].squeeze(-1)
        probabilities = torch.sigmoid(logits).cpu().tolist()
        labels = batch["labels"][:, label_index].int().tolist()
        rows.extend(
            {
                "patient_id": str(patient),
                "study_id": str(study),
                "y_true": int(label),
                "y_prob": float(probability),
            }
            for patient, study, label, probability in zip(
                batch["patient_id"], batch["study_id"], labels, probabilities
            )
        )
    return rows


@torch.no_grad()
def _counterfactual_rows(model, loader, primary, label_index, context, task, seed, threshold):
    model.eval()
    specification = str(task["input_counterfactual"])
    region = specification.removeprefix("remove_")
    policy = _policy(task)
    local: list[dict[str, Any]] = []
    for batch in loader:
        moved = move_to_device(batch, context.device)
        volume = moved["volume"]
        masks = moved["masks"]
        original_logits = model(volume, masks)["logits"][primary].squeeze(-1)
        random_metadata: list[dict[str, Any] | None]
        if region == "random":
            matched_region = str(task.get("matched_region", "pa"))
            if "random" in masks:
                selected_mask = masks["random"]
                random_metadata = [
                    {
                        "method": "precomputed_roi01_matched_random_control",
                        "roi_id": "ROI8",
                        "control_for": str(task.get("random_control_for", "ROI4")),
                        "source": "original_ctpa_roi_manifest",
                    }
                    for _ in range(volume.shape[0])
                ]
            else:
                raise ValueError(
                    "random counterfactual requires the precomputed ROI8 mask from "
                    "roi_manifest.csv; runtime random controls are prohibited"
                )
        else:
            selected_mask = masks[region]
            random_metadata = [None] * volume.shape[0]
        counterfactual_volume = remove_roi(volume, selected_mask, policy=policy)
        # The same frozen model and the original masks are deliberately reused here.
        counterfactual_logits = model(counterfactual_volume, masks)["logits"][primary].squeeze(-1)
        original_probability = torch.sigmoid(original_logits).cpu()
        counterfactual_probability = torch.sigmoid(counterfactual_logits).cpu()
        target = batch["labels"][:, label_index].int().tolist()
        flattened_mask = (selected_mask > 0.5).reshape(selected_mask.shape[0], -1)
        body = masks.get("body")
        flattened_body = (body > 0.5).reshape(body.shape[0], -1) if body is not None else None
        for index, (patient, study) in enumerate(zip(batch["patient_id"], batch["study_id"])):
            voxels = int(flattened_mask[index].sum().cpu())
            body_voxels = int(flattened_body[index].sum().cpu()) if flattened_body is not None else None
            original = float(original_probability[index])
            changed = float(counterfactual_probability[index])
            local.append(
                {
                    "patient_id": str(patient),
                    "study_id": str(study),
                    "condition": specification,
                    "original_probability": original,
                    "counterfactual_probability": changed,
                    "delta_probability": changed - original,
                    "original_predicted_class": int(original >= threshold),
                    "counterfactual_predicted_class": int(changed >= threshold),
                    "predicted_class": int(changed >= threshold),
                    "target": int(target[index]),
                    "roi_volume": voxels,
                    "roi_volume_unit": "voxels",
                    "roi_fraction_of_body": (
                        float(voxels / body_voxels) if body_voxels else None
                    ),
                    "random_control_metadata": (
                        json.dumps(random_metadata[index], sort_keys=True)
                        if random_metadata[index] is not None
                        else None
                    ),
                    "segmentation_rerun": False,
                    "original_masks_reused": True,
                }
            )
    return local


def main() -> int:
    parser = base_parser("Frozen remove-ROI counterfactual inference")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--patient-id", dest="patient_ids", action="append", metavar="ID")
    selection.add_argument("--max-cases", type=int)
    selection.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    if args.resume:
        raise SystemExit("--resume is not implemented for counterfactual inference")
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    config = resolve_cli_config(args)
    if config["experiment"]["stage"] != "counterfactual":
        raise SystemExit("counterfactual.py requires experiment.stage=counterfactual")
    smoke_scope = not bool(args.allow_full)
    if smoke_scope:
        config["resume"] = True
    context = initialize_distributed(str(config["compute"].get("distributed_backend") or "") or None)
    try:
        paths = ProjectPaths.resolve(config)
        preflight = rank_zero_call(context, lambda: require_preflight(config, paths))
        if smoke_scope:
            base_experiment_id = str(config["experiment"]["id"])
            scope = {
                "mode": "patient" if args.patient_ids else "limited",
                "patient_ids": list(args.patient_ids or ()),
                "max_cases": args.max_cases,
            }
            digest = hashlib.sha256(
                json.dumps(scope, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            config["experiment"]["id"] = f"{base_experiment_id}__SMOKE_{digest}"
            config["experiment"]["base_experiment_id"] = base_experiment_id
            config["run_scope"] = scope
            config = validate_config(
                {key: value for key, value in config.items() if key != "config_hash"}
            )
        seed = int(config.get("seed", 42))
        seed_everything(seed + context.rank)
        manager = OutputManager(paths)
        experiment_id = str(config["experiment"]["id"])

        def prepare() -> Path:
            destination = manager.prepare(
                "counterfactual", experiment_id, overwrite=bool(args.overwrite)
            )
            manager.write_config(destination, {**config, "resolved_paths": paths.as_dict()})
            return destination

        run_dir = rank_zero_call(context, prepare)
        checkpoint = Path(str(config["lineage"]["source_checkpoint"]))
        model, _ = build_task_model(config)
        load_checkpoint(checkpoint, model=model, strict=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        model = wrap_ddp(model, context)
        task = dict(config.get("task") or {})
        primary = str(task.get("primary_target", "pe_present"))
        labels = list((config.get("data") or {}).get("label_columns") or ())
        label_index = labels.index(primary)

        validation_loader, validation_expected = _loader(config, paths, "validation", context)
        validation_local = _original_prediction_rows(
            model, validation_loader, primary, label_index, context
        )
        validation_rows = gather_prediction_rows(
            validation_local, context, expected_ids=validation_expected
        )
        if context.is_main:
            threshold = select_threshold_on_validation(
                [row["y_true"] for row in validation_rows],
                [row["y_prob"] for row in validation_rows],
                method=str((config.get("evaluation") or {}).get("threshold_method", "youden")),
            )
        else:
            threshold = None
        if context.distributed:
            shared = [threshold]
            dist.broadcast_object_list(shared, src=0)
            threshold = float(shared[0])

        test_loader, test_expected = _loader(
            config,
            paths,
            "test",
            context,
            patient_ids=args.patient_ids,
            maximum=args.max_cases,
        )
        local = _counterfactual_rows(
            model, test_loader, primary, label_index, context, task, seed, float(threshold)
        )
        rows = gather_prediction_rows(local, context, expected_ids=test_expected)
        if context.is_main:
            original_rows = [
                {
                    "patient_id": row["patient_id"], "study_id": row["study_id"],
                    "y_true": row["target"], "y_prob": row["original_probability"],
                }
                for row in rows
            ]
            changed_rows = [
                {
                    "patient_id": row["patient_id"], "study_id": row["study_id"],
                    "y_true": row["target"], "y_prob": row["counterfactual_probability"],
                }
                for row in rows
            ]
            samples = int((config.get("evaluation") or {}).get("bootstrap_samples", 2000))
            confidence = float((config.get("evaluation") or {}).get("confidence", 0.95))
            patient_count = len({row["patient_id"] for row in rows})
            point = {
                "original": binary_classification_metrics(
                    [row["y_true"] for row in original_rows],
                    [row["y_prob"] for row in original_rows],
                    float(threshold),
                ),
                "counterfactual": binary_classification_metrics(
                    [row["y_true"] for row in changed_rows],
                    [row["y_prob"] for row in changed_rows],
                    float(threshold),
                ),
            }
            bootstrap = None
            paired = None
            if patient_count >= 2:
                from source.metrics.paired import paired_patient_bootstrap

                bootstrap = {
                    "original": bootstrap_binary_predictions(
                        original_rows, float(threshold), n_bootstrap=samples,
                        confidence=confidence, seed=seed,
                    ),
                    "counterfactual": bootstrap_binary_predictions(
                        changed_rows, float(threshold), n_bootstrap=samples,
                        confidence=confidence, seed=seed,
                    ),
                }
                paired = paired_patient_bootstrap(
                    original_rows,
                    changed_rows,
                    lambda items: {
                        key: value
                        for key, value in binary_classification_metrics(
                            [item["y_true"] for item in items],
                            [item["y_prob"] for item in items],
                            float(threshold),
                        ).items()
                        if key != "threshold"
                    },
                    n_bootstrap=samples,
                    confidence=confidence,
                    seed=seed,
                )
                write_parquet_atomic(
                    [{"metric": name, **values} for name, values in paired.items()],
                    run_dir / "paired_bootstrap_metrics.parquet",
                )
            deltas = np.asarray([row["delta_probability"] for row in rows], dtype=float)
            evaluation = {
                "primary_target": primary,
                "threshold": float(threshold),
                "threshold_source": "original_validation_predictions",
                "condition": task["input_counterfactual"],
                "frozen_model": True,
                "segmentation_rerun": False,
                "original_masks_reused": True,
                "patients": patient_count,
                "studies": len(rows),
                "delta_probability": {
                    "mean": float(deltas.mean()),
                    "median": float(np.median(deltas)),
                    "mean_absolute": float(np.abs(deltas).mean()),
                },
                "point_metrics": point,
                "bootstrap_metrics": bootstrap,
                "paired_counterfactual_vs_original": paired,
                "bootstrap_unavailable_reason": (
                    "smoke selection has fewer than two patients" if bootstrap is None else None
                ),
            }
            write_parquet_atomic(rows, run_dir / "counterfactual_predictions.parquet")
            source_hash = checkpoint_sha256(checkpoint)
            result = compact_result(
                config,
                status="completed",
                data={
                    "split_patients": (preflight.manifest or {}).get("split_patients", {}),
                    "split_studies": (preflight.manifest or {}).get("split_studies", {}),
                    "evaluated_patient_ids": sorted(
                        {str(row["patient_id"]) for row in rows}
                    ),
                },
                model={
                    "frozen": True,
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": source_hash,
                },
                compute={
                    "strategy": config["compute"]["strategy"],
                    "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                },
                evaluation=evaluation,
                reproducibility=environment_report(paths.code_root),
            )
            result["lineage"]["source_checkpoint_hash"] = source_hash
            manager.write_result(run_dir, result)
            print(f"COUNTERFACTUAL COMPLETED | {experiment_id} | Saved: {run_dir}")
        context.barrier()
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
