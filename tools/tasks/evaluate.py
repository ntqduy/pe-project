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
from source.data.manifests import read_rows
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.gather import gather_prediction_rows
from source.distributed.setup import initialize_distributed, rank_zero_call, wrap_ddp
from source.engine.checkpoint import checkpoint_sha256, load_checkpoint
from source.engine.experiment import (
    OutputManager,
    atomic_write_json,
    run_output_id,
    task_epoch_bundle,
)
from source.engine.factory import build_task_model
from source.engine.task_artifacts import (
    cleanup_task_run,
    latest_epoch_directory,
    preview_log_lines,
    write_backbone_previews,
)
from source.engine.trainer import move_to_device
from source.metrics.bootstrap import patient_bootstrap
from source.metrics.calibration import DEFAULT_CALIBRATION_MIN_EVENTS
from source.metrics.result_table import (
    DEFAULT_MIN_CLASS_COUNT,
    THRESHOLD_SOURCE_LOCKED,
    calibration_reason,
    log_lines,
    metric_bundle,
    prediction_rows,
    read_prediction_file,
    result_rows,
    select_threshold,
    split_summary,
    threshold_rule,
    write_predictions_csv,
    write_result_csv,
)
from source.metrics.segmentation import segmentation_case_metrics
from source.utils.console import BAR, final_evaluation_block
from source.utils.environment import environment_report
from source.utils.logger import RunLogger
from source.utils.progress import with_progress
from source.utils.seed import seed_everything
from source.utils.workers import LOADER_WORKER_GB, resolve_workers
from tools._common import (
    base_parser,
    build_dataset,
    resolve_cli_config,
    write_csv_atomic,
    write_parquet_atomic,
)


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    """Reference predictions: epoch_<N>/predictions.csv (test rows) or a legacy parquet."""
    return read_prediction_file(path, split="test")


def _locked_external_threshold(config, paths, checkpoint: Path) -> tuple[float, Path]:
    """Read a threshold selected on the internal validation cohort.

    An external test set must never supply a validation subset for threshold tuning.
    The artifact is normally the ``result.json`` beside the source checkpoint after
    internal evaluation, but can be named explicitly for exported checkpoints.
    """
    external = dict(config.get("external_evaluation") or {})
    raw = external.get("threshold_artifact")
    artifact = Path(str(raw)) if raw else checkpoint.parent / "result.json"
    if raw and not artifact.is_absolute():
        artifact = paths.output_asset(artifact)
    if not artifact.is_file():
        raise FileNotFoundError(
            "external test requires the internal-validation result containing the locked "
            f"threshold: {artifact}"
        )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    evaluation = dict(payload.get("evaluation") or {})
    if evaluation.get("threshold") is None:
        raise ValueError(f"threshold is absent from internal evaluation artifact: {artifact}")
    if str(evaluation.get("threshold_source") or "") != "validation":
        raise ValueError(
            "external threshold artifact must record threshold_source=validation: "
            f"{artifact}"
        )
    return float(evaluation["threshold"]), artifact.resolve()


def _read_restriction(path: Path) -> set[tuple[str, str]]:
    """(patient_id, study_id) pairs every arm of a comparison must be scored on.

    Restricting each arm to one shared, pre-declared case list is what makes a baseline
    comparison valid when the baseline itself is not computable for every case -- the
    same thing INSPECT does inline with --compare_vs_pesi. A study_id column is optional;
    without it the whole patient is kept.
    """
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"restriction file has no rows: {path}")
    if "patient_id" not in rows[0]:
        raise ValueError(f"restriction file needs a patient_id column: {path}")
    has_study = "study_id" in rows[0]
    return {(str(row["patient_id"]), str(row["study_id"]) if has_study else "") for row in rows}


def _valid_label(row: dict[str, Any], column: str | None) -> bool:
    """Return whether an outcome is observable for evaluation.

    Prognosis manifests retain censored rows so the official split and cohort audit stay
    intact.  Such rows must not be converted from NaN to an integer and scored as an event.
    """
    if not column:
        return True
    raw = row.get(column)
    if raw is None or str(raw).strip() == "":
        return False
    try:
        return bool(np.isfinite(float(raw)))
    except (TypeError, ValueError):
        return False


def _loader(
    config,
    paths,
    split,
    context,
    *,
    patient_ids=None,
    maximum=None,
    restrict=None,
    evaluation_target=None,
):
    dataset = build_dataset(config, paths, split)
    rows = dataset.rows
    if restrict is not None:
        keep_pairs = {pair for pair in restrict if pair[1]}
        keep_patients = {pair[0] for pair in restrict if not pair[1]}
        selected = [
            row
            for row in rows
            if (str(row["patient_id"]), str(row["study_id"])) in keep_pairs
            or str(row["patient_id"]) in keep_patients
        ]
        if not selected:
            raise ValueError(f"restriction removed every {split} row; nothing to evaluate")
        dataset.rows = selected
        rows = selected
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
    if evaluation_target is not None:
        selected = [row for row in rows if _valid_label(row, evaluation_target)]
        if not selected:
            raise ValueError(
                f"no evaluable rows remain in {split} for target={evaluation_target!r}"
            )
        dataset.rows = selected
        rows = selected
    sampler = DistributedSampler(dataset, shuffle=False) if context.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int((config.get("evaluation") or {}).get("batch_size", 1)),
        sampler=sampler,
        shuffle=False,
        # Resolved once in main() from compute.num_workers ("auto" or a RAM-capped number);
        # the order of the rows is unchanged with workers.
        num_workers=int((config.get("compute") or {}).get("num_workers_resolved", 0)),
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
            output = model(moved)
            logits = output.get("target_logits", {}).get(primary, output["logits"])
        probability = torch.sigmoid(logits).detach().cpu().tolist()
        raw_truth = batch["labels"][:, label_index].tolist()
        valid = batch["label_valid"][:, label_index].bool().tolist()
        truth = [int(value) if is_valid else 0 for value, is_valid in zip(raw_truth, valid)]
        local.extend(
            {
                "patient_id": str(patient),
                "study_id": str(study),
                "y_true": int(label),
                "y_prob": float(score),
            }
            for (patient, study), label, score, is_valid in zip(
                identifiers, truth, probability, valid
            )
            if is_valid
        )
    return local


@torch.no_grad()
def _classification_rows_all_targets(model, loader, stage, targets, context, task, seed, log=None, label="inference"):
    """Run one inference pass and emit one prediction row per observable target."""
    model.eval()
    target_names = tuple(str(target) for target in targets)
    local: list[dict[str, Any]] = []
    for batch in with_progress(loader, log, label):
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
            output = model(moved["report_embedding"])
        elif stage == "diagnosis":
            output = model(moved["volume"], moved["masks"])
        else:
            output = model(moved)
        logits_by_target = output.get("target_logits")
        if not logits_by_target:
            logits_by_target = output.get("logits")
        if not isinstance(logits_by_target, dict):
            logits_by_target = {}
        for target in target_names:
            if target not in logits_by_target:
                continue
            probabilities = torch.sigmoid(logits_by_target[target].reshape(-1)).detach().cpu().tolist()
            label_index = int((loader.dataset.label_columns).index(target))
            truth_values = batch["labels"][:, label_index].tolist()
            valid_values = batch["label_valid"][:, label_index].bool().tolist()
            local.extend(
                {
                    "patient_id": str(patient),
                    "study_id": str(study),
                    "target": target,
                    "y_true": int(value),
                    "y_prob": float(probability),
                }
                for (patient, study), value, probability, valid in zip(
                    identifiers, truth_values, probabilities, valid_values
                )
                if valid
            )
    return local


def _valid_prediction_ids(rows, target: str) -> set[tuple[str, str]]:
    return {
        (str(row["patient_id"]), str(row["study_id"]))
        for row in rows
        if _valid_label(row, target)
    }


def _gather_target_rows(local_rows, target, dataset_rows, context):
    local = [dict(row) for row in local_rows if str(row.get("target")) == target]
    expected = _valid_prediction_ids(dataset_rows, target)
    return gather_prediction_rows(local, context, expected_ids=expected)


def _run_checkpoint(run_dir: Path) -> Path:
    """Resolve the canonical checkpoint after root compatibility copies are cleaned."""
    root = run_dir / "best.ckpt"
    if root.is_file():
        return root
    bundle = run_dir / "checkpoint" / "best.ckpt"
    if bundle.is_file():
        return bundle
    candidates = sorted(
        run_dir.glob("epoch_*/checkpoint/best.ckpt"),
        key=lambda path: int(path.parent.parent.name.removeprefix("epoch_")),
    )
    if not candidates:
        raise FileNotFoundError(f"no best checkpoint found under {run_dir}")
    return candidates[-1]


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
        "--restrict-to",
        type=Path,
        default=None,
        help=(
            "CSV/parquet of patient_id[,study_id] limiting evaluation to one shared case "
            "list, so every arm of a comparison is scored on identical cases. Stage 0 "
            "writes cache/<profile>/clinical/spesi_evaluable.csv for exactly this purpose."
        ),
    )
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
    external_evaluation = dict(config.get("external_evaluation") or {})
    if external_evaluation.get("test_only"):
        if args.checkpoint is None:
            raise SystemExit("test-only external evaluation requires an explicit --checkpoint")
        if (
            str(external_evaluation.get("threshold_source"))
            != "internal_validation_artifact"
            or str(external_evaluation.get("evaluation_split")) != "test"
        ):
            raise SystemExit(
                "external evaluation contract requires an internal-validation threshold "
                "artifact and evaluation_split=test"
            )
    config["resume"] = True
    context = initialize_distributed(str(config["compute"].get("distributed_backend") or "") or None)
    logger: RunLogger | None = None

    def log(message: str) -> None:
        if logger is not None:
            logger.log(message)

    try:
        paths = ProjectPaths.resolve(config)
        seed_everything(int(config.get("seed", 42)) + context.rank)
        rank_zero_call(context, lambda: require_preflight(config, paths))
        manager = OutputManager(paths)
        family = str(config["experiment"].get("family") or config["experiment"]["stage"])
        # Same folder train_task.py wrote: <id>/epoch_<training.epochs> for a trained task.
        output_id = run_output_id(config)
        run_dir = manager.run_dir(family, output_id)
        if not run_dir.exists():
            if args.checkpoint is None:
                raise FileNotFoundError(
                    f"trained run not found: {run_dir}; train with the same training.epochs "
                    "first, or provide --checkpoint for external evaluation"
                )

            def prepare() -> Path:
                destination = manager.prepare(family, output_id)
                manager.write_config(destination, {**config, "resolved_paths": paths.as_dict()})
                return destination

            run_dir = rank_zero_call(context, prepare)
        checkpoint = args.checkpoint or _run_checkpoint(run_dir)
        restriction = None if args.restrict_to is None else _read_restriction(args.restrict_to)
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
        # Human-facing artifacts (result.csv, predictions.csv, logs.txt) go into the epoch
        # bundle next to the checkpoint; a smoke-scope evaluation keeps its own directory so
        # it never overwrites the full-test tables. result.json stays in evaluation_dir.
        # A trained task run is itself the bundle; older run folders hold epoch_<N>/.
        bundle_dir = run_dir if task_epoch_bundle(config) else latest_epoch_directory(run_dir)
        if smoke_scope:
            output_dir = evaluation_dir
        else:
            output_dir = bundle_dir or run_dir
        if context.is_main:
            # Append: logs.txt already holds the training log of this run.
            logger = RunLogger(output_dir / "logs.txt")
        log(BAR)
        log(f"EVALUATION {config['experiment']['id']} | scope={json.dumps(scope, sort_keys=True)}")
        log(f"checkpoint={Path(checkpoint).resolve()}")
        log(f"output={output_dir}")
        log(BAR)
        model, _ = build_task_model(config)
        checkpoint_payload = load_checkpoint(checkpoint, model=model, strict=True)
        model = wrap_ddp(model, context)
        # Sized after the model is loaded, so the free RAM already excludes it.
        loader_workers, worker_note = resolve_workers(
            (config.get("compute") or {}).get("num_workers", 0), per_worker_gb=LOADER_WORKER_GB,
            minimum=0, maximum=16, share=context.world_size,
        )
        config.setdefault("compute", {})["num_workers_resolved"] = loader_workers
        log(f"data loader {worker_note}")
        stage = str(config["experiment"]["stage"])
        task_config = dict(config.get("task") or {})
        if stage in {"ablation", "roi_student"}:
            stage = str(task_config.get("base_stage") or "")
        evaluation_config = dict(config.get("evaluation") or {})
        table_rows: list[dict[str, Any]] = []
        prediction_table: list[dict[str, Any]] = []
        samples = int(evaluation_config.get("bootstrap_samples", 2000))
        confidence = float(evaluation_config.get("confidence", 0.95))
        seed = int(config.get("seed", 42))
        if stage in {"diagnosis", "prognosis"}:
            default_primary = "pe_present" if stage == "diagnosis" else "mortality_30d"
            primary = str((config.get("task") or {}).get("primary_target", default_primary))
            labels = list((config.get("data") or {}).get("label_columns") or ())
            if primary not in labels:
                raise ValueError(f"primary target {primary!r} is absent from data.label_columns")
            configured_targets = list((task_config.get("targets") or {}).keys())
            if stage == "diagnosis":
                target_names = [primary]
            elif configured_targets:
                target_names = [target for target in configured_targets if target in labels]
            else:
                target_names = list(labels)
            if external_evaluation.get("test_only"):
                target_names = [primary]
            if not target_names:
                target_names = [primary]

            threshold_artifact = None
            thresholds: dict[str, float] = {}
            threshold_sources: dict[str, str] = {}
            threshold_method = str(evaluation_config.get("threshold_method", "youden"))
            calibration_min_events = int(
                evaluation_config.get("calibration_min_events", DEFAULT_CALIBRATION_MIN_EVENTS)
            )
            min_class_count = int(
                evaluation_config.get("min_class_count_warning", DEFAULT_MIN_CLASS_COUNT)
            )
            validation_rows_by_target: dict[str, list[dict[str, Any]] | None] = {}
            train_rows_by_target: dict[str, list[dict[str, Any]] | None] = {}
            validation_loader = None
            if external_evaluation.get("test_only"):
                if context.is_main:
                    threshold, threshold_artifact = _locked_external_threshold(
                        config, paths, Path(checkpoint)
                    )
                    thresholds[primary] = float(threshold)
                    threshold_sources[primary] = THRESHOLD_SOURCE_LOCKED
            else:
                train_loader, _ = _loader(
                    config, paths, "train", context, restrict=restriction
                )
                train_local = _classification_rows_all_targets(
                    model, train_loader, stage, target_names, context, task_config, seed,
                    log=log, label="evaluate train",
                )
                for target in target_names:
                    train_rows_by_target[target] = _gather_target_rows(
                        train_local, target, train_loader.dataset.rows, context
                    )
                validation_loader, _ = _loader(
                    config, paths, "validation", context, restrict=restriction
                )
                validation_local = _classification_rows_all_targets(
                    model, validation_loader, stage, target_names, context, task_config, seed,
                    log=log, label="evaluate validation",
                )
                for target in target_names:
                    validation_rows_by_target[target] = _gather_target_rows(
                        validation_local, target, validation_loader.dataset.rows, context
                    )
                if context.is_main:
                    # Youden on validation; 0.5 when validation holds a single class.
                    for target in target_names:
                        thresholds[target], threshold_sources[target] = select_threshold(
                            validation_rows_by_target[target] or [], threshold_method
                        )
            if context.distributed:
                payload = [thresholds, threshold_sources]
                dist.broadcast_object_list(payload, src=0)
                thresholds = dict(payload[0])
                threshold_sources = dict(payload[1])
            for target in target_names:
                if target in thresholds:
                    log(
                        f"threshold target={target} value={thresholds[target]:.6f} "
                        f"rule={threshold_rule(threshold_sources[target], threshold_method)} "
                        "(p >= threshold -> predicted positive; applied unchanged to every split)"
                    )

            test_loader, _ = _loader(
                config,
                paths,
                "test",
                context,
                patient_ids=args.patient_ids,
                maximum=args.max_cases,
                restrict=restriction,
            )
            test_local = _classification_rows_all_targets(
                model, test_loader, stage, target_names, context, task_config, seed,
                log=log, label="evaluate test",
            )
            rows_by_target: dict[str, list[dict[str, Any]] | None] = {}
            for target in target_names:
                rows_by_target[target] = _gather_target_rows(
                    test_local, target, test_loader.dataset.rows, context
                )
            rows: list[dict[str, Any]] | None = [] if context.is_main else None
            target_results: dict[str, dict[str, Any]] = {}
            if context.is_main:
                experiment_name = str(config["experiment"]["id"])
                for target in target_names:
                    target_rows = [dict(row) for row in (rows_by_target[target] or [])]
                    threshold = float(thresholds[target])
                    for row in target_rows:
                        row["y_pred"] = int(row["y_prob"] >= threshold)
                    rows.extend(target_rows)
                    metrics, bootstrap_reason = metric_bundle(
                        target_rows,
                        stage,
                        threshold,
                        with_ci=True,
                        samples=samples,
                        confidence=confidence,
                        seed=seed,
                        calibration_min_events=calibration_min_events,
                    )
                    validation_rows = validation_rows_by_target.get(target) or []
                    train_rows = train_rows_by_target.get(target) or []
                    rows_by_split = {"train": train_rows, "validation": validation_rows, "test": target_rows}
                    # Bootstrap CI is computed on the test split only, by protocol.
                    split_metrics: dict[str, Any] = {
                        "train": metric_bundle(
                            train_rows, stage, threshold, with_ci=False,
                            calibration_min_events=calibration_min_events,
                        )[0],
                        "validation": metric_bundle(
                            validation_rows, stage, threshold, with_ci=False,
                            calibration_min_events=calibration_min_events,
                        )[0],
                        "test": metrics,
                    }
                    table_splits: dict[str, dict[str, Any]] = {}
                    for split, items in rows_by_split.items():
                        summary = split_summary(items, threshold)
                        split_metrics[f"{split}_rows"] = summary["n_studies"]
                        split_metrics[f"{split}_patients"] = summary["n_patients"]
                        split_metrics[f"{split}_positives"] = summary["n_pos"]
                        split_metrics[f"{split}_negatives"] = summary["n_neg"]
                        if not items and split != "test":
                            continue  # test-only external runs score no train/validation rows
                        for line in log_lines(
                            target, split, summary, threshold=threshold, min_class_count=min_class_count
                        ):
                            log(line)
                        table_splits[split] = {
                            "metrics": split_metrics[split],
                            "summary": summary,
                            "ci_reason": bootstrap_reason if split == "test" else None,
                            "calibration_reason": (
                                calibration_reason(items, min_events=calibration_min_events)
                                if stage == "prognosis"
                                else None
                            ),
                        }
                        prediction_table.extend(prediction_rows(split, target, items, threshold))
                    table_rows.extend(
                        result_rows(
                            experiment=experiment_name,
                            target=target,
                            splits=table_splits,
                            threshold=threshold,
                            threshold_source=threshold_sources[target],
                            stage=stage,
                            threshold_method=threshold_method,
                            min_class_count=min_class_count,
                        )
                    )
                    target_result: dict[str, Any] = {
                        "target": target,
                        "threshold": threshold,
                        "threshold_source": threshold_sources[target],
                        "threshold_rule": threshold_rule(threshold_sources[target], threshold_method),
                        "metrics": metrics,
                        "split_metrics": split_metrics,
                        "validation_patients": len({row["patient_id"] for row in validation_rows}),
                        "evaluated_patients": len({row["patient_id"] for row in target_rows}),
                        "bootstrap": {"unit": "patient", "samples": samples, "confidence": confidence},
                        "status": "ok" if target_rows else "unavailable",
                        "unavailable_reason": bootstrap_reason,
                    }
                    if stage == "prognosis" and target_rows:
                        from source.metrics.calibration import calibration_curve_points

                        # Kept in result.json; the per-target parquet copies were deleted by
                        # the run cleanup and are no longer written.
                        target_result["calibration_curve"] = calibration_curve_points(
                            [row["y_true"] for row in target_rows],
                            [row["y_prob"] for row in target_rows],
                            bins=int(evaluation_config.get("calibration_bins", 10)),
                            strategy=str(evaluation_config.get("calibration_strategy", "quantile")),
                        )
                    target_results[target] = target_result
                primary_result = target_results[primary]
                primary_rows = [row for row in rows if str(row.get("target")) == primary]
                result_evaluation: dict[str, Any] = {
                    "primary_target": primary,
                    "target_order": target_names,
                    "cohort": (config.get("data") or {}).get("cohort") or stage,
                    "threshold": primary_result["threshold"],
                    "threshold_source": primary_result["threshold_source"],
                    "threshold_rule": primary_result["threshold_rule"],
                    "threshold_method": threshold_method,
                    "threshold_artifact": (
                        str(threshold_artifact) if threshold_artifact is not None else None
                    ),
                    "bootstrap": {"unit": "patient", "samples": samples, "confidence": confidence},
                    "metrics": primary_result["metrics"],
                    "split_metrics": {
                        target: target_result["split_metrics"]
                        for target, target_result in target_results.items()
                    },
                    "evaluated_patients": primary_result["evaluated_patients"],
                    "bootstrap_unavailable_reason": primary_result["unavailable_reason"],
                    "targets": target_results,
                }
                if args.reference_predictions is not None:
                    from source.metrics.classification import binary_classification_metrics
                    from source.metrics.paired import paired_patient_bootstrap

                    reference_rows = _read_prediction_rows(args.reference_predictions)
                    if any("target" in item for item in reference_rows):
                        reference_rows = [
                            item for item in reference_rows if str(item.get("target")) == primary
                        ]
                    paired_threshold = float(primary_result["threshold"])

                    def _paired_metric(items: list[dict[str, Any]]) -> dict[str, float]:
                        computed = binary_classification_metrics(
                            [int(item["y_true"]) for item in items],
                            [float(item["y_prob"]) for item in items],
                            paired_threshold,
                        )
                        computed.pop("threshold")
                        return computed

                    paired = paired_patient_bootstrap(
                        reference_rows,
                        primary_rows,
                        _paired_metric,
                        n_bootstrap=samples,
                        confidence=confidence,
                        seed=seed,
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
                patient_ids=args.patient_ids, maximum=args.max_cases, restrict=restriction,
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
            if stage in {"diagnosis", "prognosis"}:
                write_result_csv(output_dir / "result.csv", table_rows, stage=stage)
                write_predictions_csv(output_dir / "predictions.csv", prediction_table)
            else:
                write_csv_atomic(rows, output_dir / "predictions.csv")
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
            # Keep the training summary train_task.py recorded (epochs_run, stopped_early, ...):
            # epoch_<E> names the budget, so this is where the epochs actually run are kept.
            training_summary = (result.get("evaluation") or {}).get("training")
            result["evaluation"] = (
                {**result_evaluation, "training": training_summary}
                if training_summary is not None and "training" not in result_evaluation
                else result_evaluation
            )
            result["evaluation_scope"] = scope
            result["evaluation_checkpoint"] = {
                "path": str(Path(checkpoint).resolve()),
                "sha256": checkpoint_sha256(checkpoint),
                "lineage": checkpoint_payload.get("lineage"),
                "load_report": checkpoint_payload.get("load_report"),
            }
            atomic_write_json(result_path, result)
            if (
                stage in {"diagnosis", "prognosis"}
                and not external_evaluation.get("test_only")
                and validation_loader is not None
            ):
                if bundle_dir is not None:
                    preview_report = write_backbone_previews(
                        model,
                        validation_loader.dataset,
                        config,
                        context.device,
                        bundle_dir / "preview",
                        maximum_patients=5,
                        # Same validation-selected cutoff as result.csv, so each preview can say
                        # whether the model is right for that patient.
                        threshold=thresholds.get(primary),
                        threshold_rule=threshold_rule(threshold_sources.get(primary), threshold_method),
                        # The validation probabilities scored above: shown in the header and
                        # checked against the preview's own forward pass.
                        reference_probabilities={
                            (str(row["patient_id"]), str(row["study_id"])): float(row["y_prob"])
                            for row in validation_rows_by_target.get(primary) or ()
                        },
                        checkpoint=Path(checkpoint).resolve(),
                        checkpoint_sha256=result["evaluation_checkpoint"]["sha256"],
                    )
                    for line in preview_log_lines(preview_report, bundle_dir / "preview"):
                        log(line)
            if stage in {"diagnosis", "prognosis"}:
                from source.metrics.reporting import stard_ai_checklist, tripod_ai_checklist

                checklist = (
                    stard_ai_checklist(config, result_evaluation)
                    if stage == "diagnosis"
                    else tripod_ai_checklist(config, result_evaluation)
                )
                atomic_write_json(evaluation_dir / "reporting_checklist.json", checklist)
            cleanup_task_run(run_dir)
            log(
                f"written: {output_dir / 'result.csv'} | {output_dir / 'predictions.csv'} | {result_path}"
                if stage in {"diagnosis", "prognosis"}
                else f"written: {output_dir / 'predictions.csv'} | {result_path}"
            )
            log(final_evaluation_block(str(config["experiment"]["id"]), result_evaluation, output_dir))
            log("evaluation status=finished")
        context.barrier()
        return 0
    except Exception as exc:
        if logger is not None:
            logger.exception("evaluation status=failed", exc)  # noqa: PLE1205, TRY401 - RunLogger.exception(message, error)
        raise
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
