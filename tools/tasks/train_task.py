from __future__ import annotations

import sys
import json
from collections.abc import Mapping
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.setup import initialize_distributed, rank_zero_call, wrap_ddp
from source.engine.checkpoint import load_checkpoint
from source.engine.experiment import OutputManager, compact_result, run_output_id
from source.engine.factory import build_task_model
from source.engine.task_steps import diagnosis_evaluation_targets, task_loss_step
from source.engine.schedulers import build_scheduler
from source.engine.trainer import Trainer, move_to_device, resolve_precision
from source.engine.task_artifacts import (
    best_epoch,
    cleanup_task_run,
    preview_log_lines,
    refresh_epoch_log,
    write_backbone_previews,
    write_training_artifacts,
)
from source.profiling.model_profile import profile_model
from source.components.peft.freeze import trainable_parameter_summary
from source.utils.console import experiment_header
from source.utils.environment import environment_report
from source.utils.seed import loader_seeding, seed_everything
from source.utils.logger import RunLogger
from source.utils.progress import with_progress
from source.utils.workers import LOADER_WORKER_GB, resolve_workers
from tools._common import (
    base_parser,
    build_dataset,
    build_training_lineage,
    resolve_cli_config,
)


@torch.no_grad()
def _fit_feature_standardizer(model, dataset, context, *, batch_size, workers):
    """Fit the classifier's pooled-feature z-score on training rows (frozen CT-FM features).

    Each rank pools a disjoint shard and the sums are all-reduced inside the model, so every
    rank fits identical buffers before DDP wraps the model. None when the head does not
    standardize its inputs.
    """
    fit = getattr(model, "fit_input_standardizer", None)
    if not callable(fit):
        return None
    shard = (
        Subset(dataset, range(context.rank, len(dataset), context.world_size))
        if context.distributed
        else dataset
    )
    loader = DataLoader(shard, batch_size=batch_size, shuffle=False, num_workers=workers)
    return fit(loader, context.device, context.distributed)


@torch.no_grad()
def _auc_rows(model, loader, config, context, log=None, label="epoch AUROC pass"):
    """Collect configured target probabilities for one epoch's AUC curves."""
    task = dict(config.get("task") or {})
    data = dict(config.get("data") or {})
    stage = str((config.get("experiment") or {}).get("stage") or "")
    primary = str(task.get("primary_target") or "pe_present")
    columns = list(data.get("label_columns") or ())
    if primary not in columns:
        raise ValueError(f"primary target {primary!r} is not present in data.label_columns")
    configured_targets = list((task.get("targets") or {}).keys())
    if stage == "diagnosis":
        # Primary first; multitask native heads get their own AUROC columns.
        targets = diagnosis_evaluation_targets(task, columns, primary)
    elif configured_targets:
        targets = [target for target in configured_targets if target in columns]
    else:
        targets = list(columns)
    model.eval()
    rows = []
    for batch in with_progress(loader, log, label):
        moved = move_to_device(batch, context.device)
        if stage == "diagnosis":
            output = model(moved["volume"], moved["masks"])
            logits_by_target = output["logits"]
        elif stage == "prognosis":
            output = model(dict(moved))
            logits_by_target = output.get("target_logits") or {}
        else:
            return []
        for target in targets:
            if target not in logits_by_target:
                continue
            label_index = columns.index(target)
            labels = moved["labels"][:, label_index].reshape(-1)
            valid = moved["label_valid"][:, label_index].bool().reshape(-1)
            probabilities = torch.sigmoid(logits_by_target[target].reshape(-1))
            for index, (patient, study) in enumerate(zip(batch["patient_id"], batch["study_id"])):
                if bool(valid[index]):
                    rows.append({
                        "patient_id": str(patient), "study_id": str(study), "target": target,
                        "y_true": int(labels[index].detach().cpu()),
                        "y_prob": float(probabilities[index].detach().cpu()),
                    })
    return rows


def _epoch_auc_metrics(
    model, train_loader, validation_loader, config, context, log=None, include_train=True,
) -> Mapping[str, float]:
    from source.distributed.gather import gather_prediction_rows

    # record_epoch_auc re-reads the whole train and validation split every epoch; a run that
    # only selects on validation AUROC (include_train=False) skips the train pass.
    train_local = (
        _auc_rows(model, train_loader, config, context, log, "epoch AUROC pass train") if include_train else []
    )
    validation_local = _auc_rows(model, validation_loader, config, context, log, "epoch AUROC pass validation")
    task = dict(config.get("task") or {})
    data = dict(config.get("data") or {})
    stage = str((config.get("experiment") or {}).get("stage") or "")
    primary = str(task.get("primary_target") or "pe_present")
    columns = list(data.get("label_columns") or ())
    configured_targets = list((task.get("targets") or {}).keys())
    if stage == "diagnosis":
        # Primary first; multitask native heads get their own AUROC columns.
        targets = diagnosis_evaluation_targets(task, columns, primary)
    elif configured_targets:
        targets = [target for target in configured_targets if target in columns]
    else:
        targets = list(columns)

    def expected(dataset, target):
        selected = set()
        for row in dataset.rows:
            raw = row.get(target)
            try:
                valid = raw is not None and str(raw).strip() != "" and torch.isfinite(
                    torch.tensor(float(raw))
                )
            except (TypeError, ValueError):
                valid = False
            if bool(valid):
                selected.add((str(row["patient_id"]), str(row["study_id"])))
        return selected

    train_rows_by_target = {}
    validation_rows_by_target = {}
    for target in targets:
        train_rows_by_target[target] = gather_prediction_rows(
            [row for row in train_local if row.get("target") == target],
            context,
            expected_ids=expected(train_loader.dataset, target),
        ) if include_train else []
        validation_rows_by_target[target] = gather_prediction_rows(
            [row for row in validation_local if row.get("target") == target],
            context,
            expected_ids=expected(validation_loader.dataset, target),
        )
    values: dict[str, float] = {}
    if context.is_main:
        from sklearn.metrics import average_precision_score, roc_auc_score

        def score(rows, function):
            truth = [int(row["y_true"]) for row in rows or []]
            probability = [float(row["y_prob"]) for row in rows or []]
            return float(function(truth, probability)) if len(set(truth)) == 2 else float("nan")

        for target in targets:
            train_rows = train_rows_by_target[target]
            validation_rows = validation_rows_by_target[target]
            values[f"train_{target}_auroc"] = score(train_rows, roc_auc_score)
            values[f"val_{target}_auroc"] = score(validation_rows, roc_auc_score)
            values[f"train_{target}_auprc"] = score(train_rows, average_precision_score)
            values[f"val_{target}_auprc"] = score(validation_rows, average_precision_score)
        # Keep the short names for the primary target so existing plotting/report readers
        # and diagnosis artifacts remain compatible.
        values["train_auroc"] = values[f"train_{primary}_auroc"]
        values["val_auroc"] = values[f"val_{primary}_auroc"]
        values["train_auprc"] = values[f"train_{primary}_auprc"]
        values["val_auprc"] = values[f"val_{primary}_auprc"]
    if context.distributed:
        payload = [values]
        torch.distributed.broadcast_object_list(payload, src=0)
        values = dict(payload[0])
    return values


def main() -> int:
    parser = base_parser("Train Diagnosis or Prognosis with the shared engine")
    args = parser.parse_args()
    if args.resume:
        raise SystemExit(
            "--resume is not implemented for task training; "
            "refusing to merge a fresh run into old output"
        )
    config = resolve_cli_config(args)
    external_evaluation = dict(config.get("external_evaluation") or {})
    if external_evaluation.get("test_only") or external_evaluation.get("prohibit_training"):
        raise SystemExit(
            "this is a test-only external-evaluation config; use tools/tasks/evaluate.py "
            "with --checkpoint, never train_task.py"
        )
    context = initialize_distributed(str(config["compute"].get("distributed_backend") or "") or None)
    run_logger: RunLogger | None = None
    try:
        paths = ProjectPaths.resolve(config)
        preflight = rank_zero_call(context, lambda: require_preflight(config, paths))
        manager = OutputManager(paths)
        family = str(config["experiment"].get("family") or config["experiment"]["stage"])
        experiment_id = str(config["experiment"]["id"])
        # <id>/epoch_<training.epochs>: the whole run, metadata included, lives in its bundle.
        output_id = run_output_id(config)
        run_dir = manager.run_dir(family, output_id)

        def prepare() -> Path:
            destination = manager.prepare(
                family,
                output_id,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            manager.write_config(destination, {**config, "resolved_paths": paths.as_dict()})
            return destination

        run_dir = rank_zero_call(context, prepare)
        if context.is_main:
            run_logger = RunLogger(run_dir / "logs" / "run.log")
            run_logger.log(experiment_header(config, run_dir, preflight.manifest))
        else:
            run_logger = None
        # Same seed on every rank until DDP wraps the model: model construction and the
        # feature-standardizer fit below must see identical initial weights on all ranks.
        seed_everything(int(config["seed"]))
        train_data = build_dataset(config, paths, "train")
        validation_data = build_dataset(config, paths, "validation")
        if run_logger is not None:
            for split_name, dataset in (("train", train_data), ("validation", validation_data)):
                dropped = getattr(dataset, "dropped_unlabeled_rows", ())
                if dropped:
                    run_logger.log(
                        f"prognosis_unlabeled_rows_skipped split={split_name} count={len(dropped)}"
                    )
                preload = getattr(dataset, "preload_report", {})
                if preload.get("enabled"):
                    run_logger.log(f"preload_inputs split={split_name} {json.dumps(preload, sort_keys=True)}")
        # compute.num_workers: "auto" or a number, capped by the RAM free on this machine.
        workers, worker_note = resolve_workers(
            config["compute"].get("num_workers", 0), per_worker_gb=LOADER_WORKER_GB,
            minimum=0, maximum=16, share=context.world_size,
        )
        if run_logger is not None:
            run_logger.log(f"data loader {worker_note}")
        batch_size = int((config.get("training") or {}).get("batch_size", 1))
        train_sampler = (
            DistributedSampler(train_data, shuffle=True, seed=int(config["seed"]))
            if context.distributed
            else None
        )
        validation_sampler = (
            DistributedSampler(validation_data, shuffle=False) if context.distributed else None
        )
        # Seeded shuffle order and per-worker python/numpy streams (source/utils/seed.py).
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=workers,
            **loader_seeding(int(config["seed"]) + context.rank),
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            sampler=validation_sampler,
            num_workers=workers,
            **loader_seeding(int(config["seed"]) + context.rank),
        )
        model, peft_report = build_task_model(config)
        pretrained_report = getattr(getattr(model, "image_encoder", None), "pretrained_report", None)
        if run_logger is not None and pretrained_report is not None:
            from source.model.base import format_pretrained_report

            encoder_name = getattr(model.image_encoder, "backbone_name", config["model"].get("backbone"))
            run_logger.log(format_pretrained_report(str(encoder_name), pretrained_report))
            run_logger.log("pretrained_weights=" + json.dumps(pretrained_report, sort_keys=True, default=str))
        if run_logger is not None:
            run_logger.log(
                "finetuning="
                + json.dumps(
                    {
                        "strategy": (config.get("finetuning") or {}).get("strategy")
                        or (config.get("peft") or {}).get("method"),
                        "peft": peft_report,
                        "parameters": trainable_parameter_summary(model),
                    },
                    sort_keys=True,
                    default=str,
                )
            )
        source_checkpoint = (config.get("lineage") or {}).get("source_checkpoint")
        if source_checkpoint:
            raise ValueError(
                "lineage.source_checkpoint is set, but runs no longer start from another task's "
                "checkpoint; every arm starts from its public weights (model.pretrained)"
            )
        feature_standardization = _fit_feature_standardizer(
            model, train_data, context, batch_size=batch_size, workers=workers
        )
        if run_logger is not None and feature_standardization is not None:
            run_logger.log(
                "feature_standardization="
                + json.dumps(feature_standardization, sort_keys=True, default=str)
            )
            if feature_standardization["identity_branches"]:
                run_logger.log(
                    "WARNING feature_standardization skipped for "
                    + ",".join(feature_standardization["identity_branches"])
                    + ": fewer than 2 training rows, so these branches enter the adapters "
                    "unstandardized"
                )
        # Encoders whose graph changes per step (PENet's stochastic depth) declare it.
        unused = bool(getattr(getattr(model, "image_encoder", None), "ddp_find_unused_parameters", False))
        model = wrap_ddp(model, context, **({"find_unused_parameters": True} if unused else {}))
        # Per-rank stream from here on, so augmentation and dropout differ across ranks.
        seed_everything(int(config["seed"]) + context.rank)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("training has no trainable parameters")
        training = dict(config.get("training") or {})
        selection = str(training.get("selection_metric") or "val_loss")
        if selection not in {"val_loss", "val_auroc"}:
            raise ValueError(f"training.selection_metric must be val_loss or val_auroc, got {selection!r}")
        record_epoch_auc = bool(training.get("record_epoch_auc", True))
        configured_precision = str(config["compute"].get("precision", "fp32"))
        precision = resolve_precision(configured_precision, context.device)
        if run_logger is not None:
            accumulation = int(training.get("gradient_accumulation", 1))
            if context.device.type == "cuda":
                properties = torch.cuda.get_device_properties(context.device)
                device_text = f"cuda:{context.device.index} {properties.name} {properties.total_memory / 1024**3:.1f} GB"
            else:
                device_text = "cpu"
            model_config = dict(config.get("model") or {})
            # 3-D arms keep the flag on model, slice-MIL arms on model.mil; unset = builder default.
            checkpointing = model_config.get(
                "gradient_checkpointing", (model_config.get("mil") or {}).get("gradient_checkpointing", "default")
            )
            run_logger.log(
                f"compute device={device_text} world_size={context.world_size} "
                f"precision={precision} (configured {configured_precision})"
            )
            run_logger.log(
                f"batch micro_batch={batch_size} gradient_accumulation={accumulation} "
                f"effective_batch={batch_size * accumulation * context.world_size} "
                f"gradient_checkpointing={checkpointing} selection={selection} "
                f"early_stopping_patience={training.get('early_stopping_patience')}"
            )
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        reproducibility = environment_report(paths.code_root)
        lineage = build_training_lineage(
            config,
            paths,
            code_commit=reproducibility["git_commit"],
            source_checkpoint=source_checkpoint,
            overrides={"config_path": str((run_dir / "resolved_config.yaml").resolve())},
        )
        loss_step = task_loss_step(config)
        trainer = Trainer(
            model,
            optimizer,
            loss_step,
            context,
            run_dir,
            lineage,
            scheduler=build_scheduler(optimizer, training),
            precision=precision,
            early_stopping_patience=training.get("early_stopping_patience"),
            accumulation_steps=int(training.get("gradient_accumulation", 1)),
            selection_metric="val_auroc" if selection == "val_auroc" else None,
        )
        epoch_metrics_fn = None
        if record_epoch_auc or selection == "val_auroc":
            epoch_metrics_fn = lambda current_model, train, validation, current_context: _epoch_auc_metrics(
                current_model, train, validation, config, current_context,
                log=run_logger.log if run_logger is not None else None,
                include_train=record_epoch_auc,
            )
        training_result = trainer.fit(
            train_loader,
            validation_loader,
            int(training.get("epochs", 1)),
            epoch_metrics_fn=epoch_metrics_fn,
        )
        # Every rank holds the same rank-reduced history, so all of them fail together here.
        selected_epoch = best_epoch(training_result["history"])
        if selected_epoch is None:
            raise RuntimeError(
                "no validation-selected checkpoint: the selection metric (primary_val_metric) "
                "was non-finite in every epoch, so best.ckpt was never written; "
                f"see {run_dir / 'logs' / 'history.csv'}"
            )
        if context.is_main:
            # All task-facing artifacts are based on the validation-selected checkpoint.
            # The root files remain the compatibility API used by evaluate.py.
            load_checkpoint(run_dir / "best.ckpt", model=model, strict=True)
            epoch_artifact_dir = write_training_artifacts(
                run_dir,
                training_result,
                lineage=lineage,
                config=config,
            )
            # Runs whose evaluation explains test cases (preview.split: test -> visualize/) skip
            # this unselected validation preview, so the bundle holds one set of Grad-CAMs.
            if str((config.get("preview") or {}).get("split") or "validation") != "test":
                preview_report = write_backbone_previews(
                    model,
                    validation_data,
                    config,
                    context.device,
                    epoch_artifact_dir / "preview",
                    maximum_patients=5,
                    checkpoint=epoch_artifact_dir / "checkpoint" / "best.ckpt",
                )
                if run_logger is not None:
                    for line in preview_log_lines(preview_report, epoch_artifact_dir / "preview"):
                        run_logger.log(line)
            underlying = model.module if hasattr(model, "module") else model
            profile_batch = move_to_device(next(iter(validation_loader)), context.device)
            effective_stage = str(config["experiment"]["stage"])
            if effective_stage == "diagnosis":
                forward = lambda: underlying(profile_batch["volume"], profile_batch["masks"])
            else:
                forward = lambda: underlying(profile_batch)
            profile = profile_model(
                underlying,
                forward,
                warmup=1,
                iterations=int((config.get("profiling") or {}).get("iterations", 5)),
                batch_size=int(profile_batch["volume"].shape[0]),
            )
            training_peak = max((row["peak_vram_gb"] for row in training_result["history"]), default=0.0)
            audit = preflight.manifest or {}
            evaluation_payload = {
                "best_validation_metric": training_result["best_validation_metric"],
                "training": {
                    "epochs_configured": training_result["epochs"],
                    "epochs_run": training_result["epochs_run"],
                    "early_stopping_patience": training_result["early_stopping_patience"],
                    "stopped_early": training_result["stopped_early"],
                    "selection_metric": "validation_auroc" if selection == "val_auroc" else "negative_validation_loss",
                    "selection_direction": "maximize",
                },
                "split_strategy": (config.get("evaluation") or {}).get(
                    "split_strategy", "patient_holdout"
                ),
            }
            result = compact_result(
                config,
                status="completed",
                data={
                    "split_patients": audit.get("split_patients", {}),
                    "split_studies": audit.get("split_studies", {}),
                    "class_distribution": audit.get("class_distribution", {}),
                    "cohort": (config.get("data") or {}).get("cohort"),
                },
                model={
                    **profile,
                    "peft": peft_report,
                    "architecture": (config.get("task") or {}).get("architecture"),
                    "modalities": (config.get("task") or {}).get("modalities", ["image"]),
                    "feature_standardization": feature_standardization,
                    "pretrained_weights": pretrained_report,
                    "head": (config.get("head") or {}).get("type"),
                },
                compute={
                    "strategy": config["compute"]["strategy"],
                    "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                    "gpu_names": reproducibility["gpu_names"],
                    "precision": precision,
                    "training_time_min": training_result["training_time_min"],
                    "peak_vram_gb": training_peak,
                    "latency_ms_per_volume": profile["latency_ms_per_volume"],
                },
                evaluation=evaluation_payload,
                reproducibility=reproducibility,
            )
            # Same rule as the Trainer and best_epoch(): first epoch with the highest finite
            # metric, so a NaN epoch can never be reported as the best one.
            best_row = next(
                row for row in training_result["history"] if int(row["epoch"]) == selected_epoch
            )
            result["lineage"] = {
                **lineage,
                "epoch": int(best_row["epoch"]),
                "validation_metric": float(best_row["primary_val_metric"]),
            }
            manager.write_result(run_dir, result, split_artifacts=False)
            if run_logger is not None:
                run_logger.log(
                    f"training run={experiment_id} status=finished "
                    f"best_epoch={int(best_row['epoch'])} "
                    f"best_validation_metric={training_result['best_validation_metric']} "
                    f"(selection: {'validation AUROC' if selection == 'val_auroc' else 'negative validation loss'})"
                )
                run_logger.log(
                    f"artifacts: {epoch_artifact_dir} "
                    "(checkpoint/, history.csv, training_curves.png); "
                    "result.csv and predictions.csv are written by the evaluation step"
                )
            refresh_epoch_log(run_dir)
            cleanup_task_run(run_dir)
            print(f"TRAINING COMPLETED | {experiment_id} | Saved: {run_dir}")
        context.barrier()
        return 0
    except Exception as exc:
        if run_logger is not None:
            run_logger.exception("training status=failed", exc)  # noqa: PLE1205, TRY401 - RunLogger.exception(message, error)
        raise
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
