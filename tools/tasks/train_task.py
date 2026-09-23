from __future__ import annotations

import sys
import json
from collections.abc import Mapping
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.setup import initialize_distributed, rank_zero_call, wrap_ddp
from source.engine.checkpoint import load_checkpoint
from source.engine.experiment import OutputManager, compact_result
from source.engine.factory import build_task_model
from source.engine.task_steps import task_loss_step
from source.engine.trainer import Trainer, move_to_device
from source.engine.task_artifacts import (
    cleanup_task_run,
    refresh_epoch_log,
    write_backbone_previews,
    write_training_artifacts,
)
from source.engine.transfer import transfer_modules
from source.profiling.model_profile import profile_model
from source.components.peft.freeze import trainable_parameter_summary
from source.utils.console import experiment_header
from source.utils.environment import environment_report
from source.utils.seed import seed_everything
from source.utils.logger import RunLogger
from tools._common import (
    base_parser,
    build_dataset,
    build_training_lineage,
    resolve_cli_config,
    write_parquet_atomic,
)


def _fit_clinical_preprocessor(model, dataset, config) -> dict[str, object] | None:
    """Fit clinical imputation/normalization on training rows only."""

    columns = tuple((config.get("data") or {}).get("ehr_columns") or ())
    encoder = getattr(model, "ehr_encoder", None) or getattr(model, "clinical_encoder", None)
    if encoder is None:
        return None
    if not columns:
        raise ValueError("an enabled EHR encoder requires data.ehr_columns")
    rows = getattr(dataset, "rows", None)
    if rows is None and hasattr(dataset, "base"):
        rows = getattr(dataset.base, "rows", None)
    if not rows:
        raise ValueError("cannot fit clinical preprocessing without training rows")

    def numeric(row, column):
        value = row.get(column)
        return float("nan") if value is None or str(value).strip() == "" else float(value)

    values = torch.tensor(
        [[numeric(row, column) for column in columns] for row in rows],
        dtype=torch.float32,
    )
    encoder.fit_preprocessor(values, split="train")
    return encoder.preprocessor.export_state(columns)


@torch.no_grad()
def _auc_rows(model, loader, config, context):
    """Collect configured target probabilities for one epoch's AUC curves."""
    from source.components.roi.masks import apply_counterfactual

    task = dict(config.get("task") or {})
    data = dict(config.get("data") or {})
    stage = str((config.get("experiment") or {}).get("stage") or "")
    if stage in {"ablation", "roi_student"}:
        stage = str(task.get("base_stage") or stage)
    primary = str(task.get("primary_target") or "pe_present")
    columns = list(data.get("label_columns") or ())
    if primary not in columns:
        raise ValueError(f"primary target {primary!r} is not present in data.label_columns")
    configured_targets = list((task.get("targets") or {}).keys())
    if stage == "diagnosis":
        targets = [primary]
    elif configured_targets:
        targets = [target for target in configured_targets if target in columns]
    else:
        targets = list(columns)
    model.eval()
    rows = []
    for batch in loader:
        moved = move_to_device(batch, context.device)
        if stage == "diagnosis":
            volume = apply_counterfactual(
                moved["volume"], moved["masks"], str(task.get("input_counterfactual") or ""),
                matched_region=str(task.get("matched_region", "pa")),
                masking_policy=task.get("masking_policy", "local_mean"),
                seed=int(config.get("seed", 42)),
                patient_ids=batch.get("patient_id"), study_ids=batch.get("study_id"),
            )
            output = model(volume, moved["masks"])
            logits_by_target = output["logits"]
        elif stage == "prognosis":
            model_batch = dict(moved)
            if "volume" in model_batch:
                model_batch["volume"] = apply_counterfactual(
                    model_batch["volume"], model_batch["masks"], str(task.get("input_counterfactual") or ""),
                    matched_region=str(task.get("matched_region", "pa")),
                    masking_policy=task.get("masking_policy", "local_mean"),
                    seed=int(config.get("seed", 42)),
                    patient_ids=batch.get("patient_id"), study_ids=batch.get("study_id"),
                )
            output = model(model_batch)
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


def _epoch_auc_metrics(model, train_loader, validation_loader, config, context) -> Mapping[str, float]:
    from source.distributed.gather import gather_prediction_rows

    train_local = _auc_rows(model, train_loader, config, context)
    validation_local = _auc_rows(model, validation_loader, config, context)
    task = dict(config.get("task") or {})
    data = dict(config.get("data") or {})
    stage = str((config.get("experiment") or {}).get("stage") or "")
    if stage in {"ablation", "roi_student"}:
        stage = str(task.get("base_stage") or stage)
    primary = str(task.get("primary_target") or "pe_present")
    columns = list(data.get("label_columns") or ())
    configured_targets = list((task.get("targets") or {}).keys())
    if stage == "diagnosis":
        targets = [primary]
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
        )
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


@torch.no_grad()
def _distillation_rows(model, teacher, loader, config, context):
    from torch.nn import functional

    from source.components.roi.masks import apply_counterfactual

    task = dict(config.get("task") or {})
    distillation = dict(config.get("distillation") or {})
    primary = str(task.get("primary_target", "pe_present"))
    columns = list((config.get("data") or {}).get("label_columns") or ())
    label_index = columns.index(primary)
    temperature = float(distillation.get("temperature", 2.0))
    alpha_supervised = float(distillation.get("alpha_supervised", 1.0))
    alpha_distill = float(distillation.get("alpha_distill", 1.0))
    model.eval()
    teacher.eval()
    rows = []
    for batch in loader:
        moved = move_to_device(batch, context.device)
        student_volume = apply_counterfactual(
            moved["volume"],
            moved["masks"],
            str(task.get("input_counterfactual") or ""),
            matched_region=str(task.get("matched_region", "pa")),
            masking_policy=task.get("masking_policy", "local_mean"),
            seed=int(config.get("seed", 42)),
            patient_ids=batch.get("patient_id"),
            study_ids=batch.get("study_id"),
        )
        student = model(student_volume, moved["masks"])["logits"][primary].squeeze(-1)
        teacher_logits = teacher(moved["volume"], moved["masks"])["logits"][primary].squeeze(-1)
        truth = moved["labels"][:, label_index].float()
        valid = moved["label_valid"][:, label_index].bool()
        gt = functional.binary_cross_entropy_with_logits(student, truth, reduction="none")
        soft = torch.sigmoid(teacher_logits.detach() / temperature)
        kd = functional.binary_cross_entropy_with_logits(
            student / temperature, soft, reduction="none"
        ) * temperature**2
        total = alpha_supervised * gt + alpha_distill * kd
        for index, (patient, study) in enumerate(zip(batch["patient_id"], batch["study_id"])):
            if not bool(valid[index]):
                continue
            rows.append(
                {
                    "patient_id": str(patient),
                    "study_id": str(study),
                    "target": float(truth[index].cpu()),
                    "teacher_logit": float(teacher_logits[index].cpu()),
                    "student_logit": float(student[index].cpu()),
                    "gt_loss": float(gt[index].cpu()),
                    "kd_loss": float(kd[index].cpu()),
                    "total_loss": float(total[index].cpu()),
                    "temperature": temperature,
                    "lambda_kd": alpha_distill,
                    "teacher_gradients_enabled": False,
                }
            )
    return rows


def main() -> int:
    parser = base_parser("Train Diagnosis, Prognosis, or Contour with the shared engine")
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
        output_id = str(config["experiment"].get("output_id") or experiment_id)
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
        seed_everything(int(config["seed"]) + context.rank)
        train_data = build_dataset(config, paths, "train")
        validation_data = build_dataset(config, paths, "validation")
        if run_logger is not None:
            for split_name, dataset in (("train", train_data), ("validation", validation_data)):
                dropped = getattr(dataset, "dropped_unlabeled_rows", ())
                if dropped:
                    run_logger.log(
                        f"prognosis_unlabeled_rows_skipped split={split_name} count={len(dropped)}"
                    )
        workers = int(config["compute"].get("num_workers", 0))
        batch_size = int((config.get("training") or {}).get("batch_size", 1))
        train_sampler = DistributedSampler(train_data, shuffle=True) if context.distributed else None
        validation_sampler = (
            DistributedSampler(validation_data, shuffle=False) if context.distributed else None
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=workers,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            sampler=validation_sampler,
            num_workers=workers,
        )
        model, peft_report = build_task_model(config)
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
        transfer_report = None
        if source_checkpoint:
            transfer_report = transfer_modules(
                model,
                source_checkpoint,
                tuple(
                    (config.get("lineage") or {}).get("transfer_modules")
                    or ("image_encoder",)
                ),
            )
            if run_logger is not None:
                run_logger.log("checkpoint_load=" + json.dumps(transfer_report, sort_keys=True, default=str))
        clinical_preprocessing = _fit_clinical_preprocessor(model, train_data, config)
        model = wrap_ddp(model, context)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("training has no trainable parameters")
        training = dict(config.get("training") or {})
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
            precision=str(config["compute"].get("precision", "fp32")),
            early_stopping_patience=training.get("early_stopping_patience"),
            accumulation_steps=int(training.get("gradient_accumulation", 1)),
        )
        epoch_metrics_fn = None
        if bool(training.get("record_epoch_auc", True)):
            epoch_metrics_fn = lambda current_model, train, validation, current_context: _epoch_auc_metrics(
                current_model, train, validation, config, current_context
            )
        training_result = trainer.fit(
            train_loader,
            validation_loader,
            int(training.get("epochs", 1)),
            epoch_metrics_fn=epoch_metrics_fn,
        )
        distillation_rows = None
        if bool((config.get("distillation") or {}).get("enabled")):
            from source.distributed.gather import gather_prediction_rows

            load_checkpoint(run_dir / "best.ckpt", model=model, strict=True)
            teacher = getattr(loss_step, "teacher", None)
            if teacher is None:
                raise RuntimeError("enabled distillation did not construct a frozen teacher")
            local_distillation = _distillation_rows(
                model, teacher.to(context.device), validation_loader, config, context
            )
            expected = {
                (str(row["patient_id"]), str(row["study_id"]))
                for row in validation_data.rows
            }
            distillation_rows = gather_prediction_rows(
                local_distillation, context, expected_ids=expected
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
                extra_parameters={
                    "dataset.profile": (config.get("data") or {}).get("profile"),
                    "dataset.manifest": (config.get("data") or {}).get("manifest"),
                    "dataset.split_policy": "official train/validation/test; no repartition",
                },
            )
            preview_report = write_backbone_previews(
                model,
                validation_data,
                config,
                context.device,
                epoch_artifact_dir / "preview",
                maximum_patients=5,
            )
            (epoch_artifact_dir / "preview" / "summary.json").write_text(
                json.dumps(preview_report, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            underlying = model.module if hasattr(model, "module") else model
            profile_batch = move_to_device(next(iter(validation_loader)), context.device)
            effective_stage = str(config["experiment"]["stage"])
            if effective_stage in {"ablation", "roi_student"}:
                effective_stage = str((config.get("task") or {}).get("base_stage"))
            if effective_stage == "diagnosis" and str((config.get("task") or {}).get("architecture", "soft_moe")) == "report_only":
                forward = lambda: underlying(profile_batch["report_embedding"])
            elif effective_stage == "diagnosis":
                forward = lambda: underlying(profile_batch["volume"], profile_batch["masks"])
            elif effective_stage == "prognosis":
                forward = lambda: underlying(profile_batch)
            else:
                forward = lambda: underlying(profile_batch["volume"])
            profile = profile_model(
                underlying,
                forward,
                warmup=1,
                iterations=int((config.get("profiling") or {}).get("iterations", 5)),
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
                    "selection_metric": "negative_validation_loss",
                    "selection_direction": "maximize",
                },
                "split_strategy": (config.get("evaluation") or {}).get(
                    "split_strategy", "patient_holdout"
                ),
            }
            if str(config["experiment"]["stage"]) == "roi_student":
                evaluation_payload["sufficiency_experiment"] = {
                    "roi_only_input": (config.get("task") or {}).get("input_counterfactual"),
                    "ground_truth_supervision": True,
                    "distillation_enabled": bool(
                        (config.get("distillation") or {}).get("enabled", False)
                    ),
                    "comparison_arm": (
                        "GT_plus_KD"
                        if (config.get("distillation") or {}).get("enabled")
                        else "GT_without_KD"
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
                    "checkpoint_load": transfer_report,
                    "architecture": (config.get("task") or {}).get("architecture"),
                    "modalities": (config.get("task") or {}).get("modalities", ["image"]),
                    "clinical_preprocessing": clinical_preprocessing,
                },
                compute={
                    "strategy": config["compute"]["strategy"],
                    "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                    "gpu_names": reproducibility["gpu_names"],
                    "precision": config["compute"].get("precision", "fp32"),
                    "training_time_min": training_result["training_time_min"],
                    "peak_vram_gb": training_peak,
                    "latency_ms_per_volume": profile["latency_ms_per_volume"],
                },
                evaluation=evaluation_payload,
                reproducibility=reproducibility,
            )
            best_row = max(
                training_result["history"],
                key=lambda row: float(row["primary_val_metric"]),
            )
            result["lineage"] = {
                **lineage,
                "epoch": int(best_row["epoch"]),
                "validation_metric": float(best_row["primary_val_metric"]),
            }
            if distillation_rows is not None:
                write_parquet_atomic(
                    distillation_rows, run_dir / "distillation_validation_predictions.parquet"
                )
                result["evaluation"]["distillation"] = {
                    "split": "validation",
                    "rows": len(distillation_rows),
                    "artifact": "distillation_validation_predictions.parquet",
                    "teacher_frozen": True,
                    "temperature": float(config["distillation"]["temperature"]),
                    "lambda_kd": float(config["distillation"]["alpha_distill"]),
                }
            manager.write_result(run_dir, result, split_artifacts=False)
            if run_logger is not None:
                run_logger.log(
                    f"training run={experiment_id} status=finished "
                    f"best_validation_metric={training_result['best_validation_metric']}"
                )
            refresh_epoch_log(run_dir, epochs_run=int(training_result["epochs_run"]))
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
