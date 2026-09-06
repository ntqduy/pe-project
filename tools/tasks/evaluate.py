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

from source.components.roi.masks import apply_counterfactual
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.gather import gather_prediction_rows
from source.distributed.setup import initialize_distributed, rank_zero_call, wrap_ddp
from source.engine.checkpoint import load_checkpoint
from source.engine.experiment import OutputManager, atomic_write_json
from source.engine.factory import build_task_model
from source.engine.trainer import move_to_device
from source.metrics.bootstrap import (
    bootstrap_binary_predictions,
    bootstrap_prognosis_predictions,
    patient_bootstrap,
)
from source.metrics.classification import select_threshold_on_validation
from source.metrics.segmentation import segmentation_case_metrics
from source.utils.console import final_evaluation_block
from source.utils.environment import environment_report
from source.utils.seed import seed_everything
from tools._common import base_parser, build_dataset, resolve_cli_config, write_parquet_atomic


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(path).to_dict(orient="records")


def _loader(config, paths, split, context, *, patient_ids=None, maximum=None):
    dataset = build_dataset(config, paths, split)
    rows = dataset.rows
    if patient_ids:
        requested = {str(value) for value in patient_ids}
        available = {str(row["patient_id"]) for row in rows}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"patient_id not found in {split}: {', '.join(missing)}")
        selected = [row for row in rows if str(row["patient_id"]) in requested]
        dataset.rows = selected
        rows = selected
    if maximum is not None:
        selected = rows[: int(maximum)]
        dataset.rows = selected
        rows = selected
    sampler = DistributedSampler(dataset, shuffle=False) if context.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int((config.get("evaluation") or {}).get("batch_size", 1)),
        sampler=sampler,
        shuffle=False,
    )
    expected = {(str(row["patient_id"]), str(row["study_id"])) for row in rows}
    return loader, expected


@torch.no_grad()
def _classification_rows(model, loader, stage, primary, label_index, context, task, seed):
    model.eval()
    local = []
    for batch in loader:
        identifiers = list(zip(batch["patient_id"], batch["study_id"]))
        moved = move_to_device(batch, context.device)
        if "volume" in moved:
            moved = dict(moved)
            moved["volume"] = apply_counterfactual(
                moved["volume"],
                moved["masks"],
                str(task.get("input_counterfactual") or ""),
                matched_region=str(task.get("matched_region", "pa")),
                masking_policy=task.get("masking_policy", "local_mean"),
                seed=seed,
                patient_ids=batch.get("patient_id"),
                study_ids=batch.get("study_id"),
            )
        if stage == "diagnosis" and str(task.get("architecture")) == "report_only":
            logits = model(moved["report_embedding"])["logits"][primary].squeeze(-1)
        elif stage == "diagnosis":
            logits = model(moved["volume"], moved["masks"])["logits"][primary].squeeze(-1)
        else:
            logits = model(moved)["logits"]
        probability = torch.sigmoid(logits).detach().cpu().tolist()
        truth = batch["labels"][:, label_index].int().tolist()
        local.extend(
            {
                "patient_id": str(patient),
                "study_id": str(study),
                "y_true": int(label),
                "y_prob": float(score),
            }
            for (patient, study), label, score in zip(identifiers, truth, probability)
        )
    return local


@torch.no_grad()
def _contour_rows(model, loader, context, tolerance):
    model.eval()
    local = []
    for batch in loader:
        identifiers = list(zip(batch["patient_id"], batch["study_id"]))
        moved = move_to_device(batch, context.device)
        prediction = (torch.sigmoid(model(moved["volume"])) >= 0.5).cpu().numpy()
        target = batch["masks"]["target"].numpy()
        for (patient, study), predicted_case, target_case in zip(identifiers, prediction, target):
            per_region = [
                segmentation_case_metrics(predicted_case[index], target_case[index], (1, 1, 1), tolerance)
                for index in range(predicted_case.shape[0])
            ]
            row = {
                "patient_id": str(patient),
                "study_id": str(study),
                "dice": float(np.mean([item["dice"] for item in per_region])),
                "nsd": float(np.mean([item["nsd"] for item in per_region])),
                "hd95": float(np.mean([item["hd95"] for item in per_region])),
            }
            for index, metrics in enumerate(per_region):
                for name, value in metrics.items():
                    row[f"{name}_region_{index}"] = value
            local.append(row)
    return local


def main() -> int:
    parser = base_parser(
        "Evaluate a trained task checkpoint with gathered patient predictions and bootstrap CI"
    )
    parser.add_argument("--checkpoint", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--patient-id", dest="patient_ids", action="append", metavar="ID")
    selection.add_argument("--max-cases", type=int)
    selection.add_argument("--allow-full", action="store_true")
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=None,
        help=(
            "predictions.parquet from a full-CT (or other reference) run, evaluated on the "
            "same patients, for a paired bootstrap delta (section 14 ROI/counterfactual comparisons)"
        ),
    )
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    config = resolve_cli_config(args)
    config["resume"] = True
    context = initialize_distributed(str(config["compute"].get("distributed_backend") or "") or None)
    try:
        paths = ProjectPaths.resolve(config)
        seed_everything(int(config.get("seed", 42)) + context.rank)
        rank_zero_call(context, lambda: require_preflight(config, paths))
        manager = OutputManager(paths)
        family = str(config["experiment"].get("family") or config["experiment"]["stage"])
        run_dir = manager.run_dir(family, str(config["experiment"]["id"]))
        if not run_dir.exists():
            if args.checkpoint is None:
                raise FileNotFoundError(
                    f"trained run not found: {run_dir}; "
                    "provide --checkpoint for external evaluation"
                )

            def prepare() -> Path:
                destination = manager.prepare(family, str(config["experiment"]["id"]))
                manager.write_config(destination, {**config, "resolved_paths": paths.as_dict()})
                return destination

            run_dir = rank_zero_call(context, prepare)
        checkpoint = args.checkpoint or run_dir / "best.ckpt"
        smoke_scope = bool(args.patient_ids or args.max_cases is not None)
        if smoke_scope:
            scope = {"patient_ids": list(args.patient_ids or ()), "max_cases": args.max_cases}
            digest = hashlib.sha256(
                json.dumps(scope, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            evaluation_dir = run_dir / "smoke" / digest
            if evaluation_dir.exists() and not args.overwrite:
                raise FileExistsError(
                    f"smoke evaluation already exists: {evaluation_dir}; use --overwrite"
                )
            evaluation_dir.mkdir(parents=True, exist_ok=True)
        else:
            scope = {"mode": "full_test"}
            evaluation_dir = run_dir
        model, _ = build_task_model(config)
        load_checkpoint(checkpoint, model=model, strict=True)
        model = wrap_ddp(model, context)
        stage = str(config["experiment"]["stage"])
        task_config = dict(config.get("task") or {})
        if stage in {"ablation", "roi_student"}:
            stage = str(task_config.get("base_stage") or "")
        evaluation_config = dict(config.get("evaluation") or {})
        samples = int(evaluation_config.get("bootstrap_samples", 2000))
        confidence = float(evaluation_config.get("confidence", 0.95))
        seed = int(config.get("seed", 42))
        if stage in {"diagnosis", "prognosis"}:
            default_primary = "pe_present" if stage == "diagnosis" else "mortality_30d"
            primary = str((config.get("task") or {}).get("primary_target", default_primary))
            labels = list((config.get("data") or {}).get("label_columns") or ())
            if primary not in labels:
                raise ValueError(f"primary target {primary!r} is absent from data.label_columns")
            label_index = labels.index(primary)
            validation_loader, validation_expected = _loader(config, paths, "validation", context)
            validation_local = _classification_rows(
                model,
                validation_loader,
                stage,
                primary,
                label_index,
                context,
                task_config,
                seed,
            )
            validation_rows = gather_prediction_rows(
                validation_local,
                context,
                expected_ids=validation_expected,
            )
            if context.is_main:
                threshold = select_threshold_on_validation(
                    [row["y_true"] for row in validation_rows],
                    [row["y_prob"] for row in validation_rows],
                    method=str(evaluation_config.get("threshold_method", "youden")),
                )
            else:
                threshold = None
            if context.distributed:
                value = [threshold]
                dist.broadcast_object_list(value, src=0)
                threshold = float(value[0])
            test_loader, test_expected = _loader(
                config, paths, "test", context,
                patient_ids=args.patient_ids, maximum=args.max_cases,
            )
            test_local = _classification_rows(
                model,
                test_loader,
                stage,
                primary,
                label_index,
                context,
                task_config,
                seed,
            )
            rows = gather_prediction_rows(test_local, context, expected_ids=test_expected)
            if context.is_main:
                for row in rows:
                    row["y_pred"] = int(row["y_prob"] >= threshold)
                bootstrap = (
                    bootstrap_prognosis_predictions
                    if stage == "prognosis"
                    else bootstrap_binary_predictions
                )
                patient_count = len({row["patient_id"] for row in rows})
                if patient_count >= 2:
                    metrics = bootstrap(
                        rows, threshold, n_bootstrap=samples, confidence=confidence, seed=seed
                    )
                else:
                    from source.metrics.classification import binary_classification_metrics
                    from source.metrics.prognosis import prognosis_metrics

                    point_fn = prognosis_metrics if stage == "prognosis" else binary_classification_metrics
                    point = point_fn(
                        [row["y_true"] for row in rows],
                        [row["y_prob"] for row in rows],
                        threshold,
                    )
                    metrics = {
                        name: {
                            "value": value, "ci_low": float("nan"),
                            "ci_high": float("nan"), "valid_replicates": 0,
                        }
                        for name, value in point.items() if name != "threshold"
                    }
                result_evaluation: dict[str, Any] = {
                    "primary_target": primary,
                    "threshold": threshold,
                    "threshold_source": "validation",
                    "bootstrap": {"unit": "patient", "samples": samples, "confidence": confidence},
                    "metrics": metrics,
                    "evaluated_patients": patient_count,
                    "bootstrap_unavailable_reason": (
                        "smoke selection has fewer than two patients" if patient_count < 2 else None
                    ),
                }
                if stage == "prognosis":
                    from source.metrics.calibration import calibration_curve_points

                    curve = calibration_curve_points(
                        [row["y_true"] for row in rows],
                        [row["y_prob"] for row in rows],
                        bins=int(evaluation_config.get("calibration_bins", 10)),
                        strategy=str(evaluation_config.get("calibration_strategy", "quantile")),
                    )
                    result_evaluation["calibration_curve"] = curve
                    write_parquet_atomic(curve, evaluation_dir / "calibration_curve.parquet")
                if args.reference_predictions is not None:
                    from source.metrics.classification import binary_classification_metrics
                    from source.metrics.paired import paired_patient_bootstrap

                    reference_rows = _read_prediction_rows(args.reference_predictions)

                    def _paired_metric(items: list[dict[str, Any]]) -> dict[str, float]:
                        computed = binary_classification_metrics(
                            [int(item["y_true"]) for item in items],
                            [float(item["y_prob"]) for item in items],
                            threshold,
                        )
                        computed.pop("threshold")
                        return computed

                    paired = paired_patient_bootstrap(
                        reference_rows, rows, _paired_metric,
                        n_bootstrap=samples, confidence=confidence, seed=seed,
                    )
                    write_parquet_atomic(
                        [{"metric": name, **values} for name, values in paired.items()],
                        evaluation_dir / "bootstrap_metrics.parquet",
                    )
                    result_evaluation["paired_vs_reference"] = {
                        "reference_predictions": str(args.reference_predictions),
                        "metrics": paired,
                    }
        elif stage == "contour":
            test_loader, test_expected = _loader(
                config, paths, "test", context,
                patient_ids=args.patient_ids, maximum=args.max_cases,
            )
            tolerance = float(evaluation_config.get("nsd_tolerance_mm", 1.0))
            local = _contour_rows(model, test_loader, context, tolerance)
            rows = gather_prediction_rows(local, context, expected_ids=test_expected)
            if context.is_main:
                metrics = patient_bootstrap(
                    rows,
                    lambda items: {
                        name: float(np.mean([row[name] for row in items]))
                        for name in ("dice", "nsd", "hd95")
                    },
                    n_bootstrap=samples,
                    confidence=confidence,
                    seed=seed,
                )
                region_indices = sorted(
                    int(key.rsplit("_", 1)[1])
                    for key in rows[0]
                    if key.startswith("dice_region_")
                )
                per_region = {
                    str(index): patient_bootstrap(
                        rows,
                        lambda items, index=index: {
                            name: float(np.mean([row[f"{name}_region_{index}"] for row in items]))
                            for name in ("dice", "nsd", "hd95")
                        },
                        n_bootstrap=samples,
                        confidence=confidence,
                        seed=seed,
                    )
                    for index in region_indices
                }
                result_evaluation = {
                    "bootstrap": {"unit": "patient", "samples": samples, "confidence": confidence},
                    "nsd_tolerance_mm": tolerance,
                    "metrics": metrics,
                    "per_region": per_region,
                }
        else:
            raise ValueError(f"evaluation does not support stage={stage}")
        if context.is_main:
            write_parquet_atomic(rows, evaluation_dir / "predictions.parquet")
            result_path = evaluation_dir / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                source_result_path = run_dir / "result.json"
                if not source_result_path.is_file():
                    source_result_path = checkpoint.parent / "result.json"
                source_result = (
                    json.loads(source_result_path.read_text(encoding="utf-8"))
                    if source_result_path.is_file()
                    else {}
                )
                result = {
                    "experiment": {
                        "id": config["experiment"]["id"],
                        "stage": config["experiment"]["stage"],
                        "status": "completed",
                        "seed": int(config.get("seed", 42)),
                    },
                    "lineage": dict(config.get("lineage") or {}),
                    "model": dict(source_result.get("model") or {}),
                    "compute": {
                        "strategy": config["compute"].get("strategy"),
                        "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                    },
                    "reproducibility": environment_report(paths.code_root),
                    "config_hash": config.get("config_hash"),
                }
            result["evaluation"] = result_evaluation
            result["evaluation_scope"] = scope
            atomic_write_json(result_path, result)
            if stage in {"diagnosis", "prognosis"}:
                from source.metrics.reporting import stard_ai_checklist, tripod_ai_checklist

                checklist = (
                    stard_ai_checklist(config, result_evaluation)
                    if stage == "diagnosis"
                    else tripod_ai_checklist(config, result_evaluation)
                )
                atomic_write_json(evaluation_dir / "reporting_checklist.json", checklist)
            print(final_evaluation_block(
                str(config["experiment"]["id"]), result_evaluation, evaluation_dir
            ))
        context.barrier()
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
