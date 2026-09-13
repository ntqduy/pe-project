from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from source.tasks.contour.losses import contour_loss
from source.tasks.diagnosis.losses import diagnosis_loss, organ_auxiliary_loss
from source.tasks.prognosis.losses import mortality_loss


def _build_frozen_teacher(config: Mapping[str, Any], teacher_checkpoint: str) -> nn.Module:
    """Build a same-architecture teacher model and load frozen weights for distillation.

    The teacher is constructed from the student's own config (ROI-student experiments
    share the base diagnosis architecture; only the input counterfactual differs), so its
    weights come entirely from ``teacher_checkpoint`` rather than the student's optimizer.
    """
    from source.distillation.teacher import FrozenTeacher
    from source.engine.checkpoint import load_checkpoint
    from source.engine.factory import build_task_model

    teacher_model, _ = build_task_model(config)
    load_checkpoint(teacher_checkpoint, model=teacher_model, strict=True)
    return FrozenTeacher(teacher_model)


def task_loss_step(config: Mapping[str, Any]):
    stage = str((config.get("experiment") or {}).get("stage"))
    data = dict(config.get("data") or {})
    task = dict(config.get("task") or {})
    if stage in {"ablation", "roi_student"}:
        stage = str(task.get("base_stage") or "")
    label_columns = tuple(data.get("label_columns") or ())
    counterfactual_seed = int(config.get("seed", 42))

    distillation = dict(config.get("distillation") or {})
    teacher: nn.Module | None = None
    if distillation.get("enabled"):
        teacher_checkpoint = distillation.get("teacher_checkpoint")
        if not teacher_checkpoint:
            raise ValueError("distillation.enabled requires distillation.teacher_checkpoint")
        if stage != "diagnosis":
            raise ValueError("distillation is only wired for the diagnosis stage")
        if str(task.get("architecture", "soft_moe")) == "report_only":
            raise ValueError("distillation is not supported for the report_only diagnosis architecture")
        teacher = _build_frozen_teacher(config, str(teacher_checkpoint))

    report_only = str(task.get("architecture", "soft_moe")) == "report_only"

    def diagnosis(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        from source.components.roi.masks import apply_counterfactual
        from source.distillation.losses import knowledge_distillation_loss

        if report_only:
            output = model(batch["report_embedding"])
        else:
            counterfactual = str(task.get("input_counterfactual") or "")
            volume = apply_counterfactual(
                batch["volume"], batch["masks"], counterfactual,
                matched_region=str(task.get("matched_region", "pa")),
                masking_policy=task.get("masking_policy", "local_mean"),
                seed=counterfactual_seed,
                patient_ids=batch.get("patient_id"),
                study_ids=batch.get("study_id"),
            )
            output = model(volume, batch["masks"])
        labels = {name: batch["labels"][:, index] for index, name in enumerate(label_columns)}
        valid = {name: batch["label_valid"][:, index] for index, name in enumerate(label_columns)}
        loss, main_report = diagnosis_loss(output["logits"], labels, valid, task.get("loss_weights"))
        metrics: dict[str, float] = {
            f"main.{name}.loss": value for name, value in main_report.items()
        }
        for name in output["logits"]:
            if name in valid:
                metrics[f"main.{name}.valid_count"] = float(valid[name].sum().detach().cpu())
        auxiliary_mapping = output.get("auxiliary_target_mapping") or {}
        if auxiliary_mapping:
            silver_targets = tuple(task.get("silver_targets") or ())
            silver_labels = {
                name: batch["silver_labels"][:, index]
                for index, name in enumerate(silver_targets)
            } if silver_targets and "silver_labels" in batch else {}
            silver_valid = {
                name: batch["silver_valid"][:, index]
                for index, name in enumerate(silver_targets)
            } if silver_targets and "silver_valid" in batch else {}
            auxiliary, auxiliary_report = organ_auxiliary_loss(
                output["auxiliary_logits"],
                auxiliary_mapping,
                {
                    "native": labels,
                    "expert_reviewed": labels,
                    "silver": silver_labels,
                },
                {
                    "native": valid,
                    "expert_reviewed": valid,
                    "silver": silver_valid,
                },
                organ_weights=task.get("auxiliary_organ_weights"),
                target_weights=task.get("auxiliary_target_weights"),
            )
            loss = loss + auxiliary
            metrics.update(auxiliary_report)
        silver_targets = tuple(task.get("silver_targets") or ())
        if silver_targets and "silver_labels" in batch and not auxiliary_mapping:
            silver_labels = {name: batch["silver_labels"][:, index] for index, name in enumerate(silver_targets)}
            silver_valid = {name: batch["silver_valid"][:, index] for index, name in enumerate(silver_targets)}
            silver_loss, _ = diagnosis_loss(
                output["logits"], silver_labels, silver_valid,
                task.get("silver_loss_weights"), allow_empty=True,
            )
            loss = loss + float(task.get("silver_loss_weight", 0.2)) * silver_loss
        if teacher is not None:
            teacher.to(batch["volume"].device)
            with torch.no_grad():
                teacher_output = teacher(batch["volume"], batch["masks"])
            primary = str(task.get("primary_target", "pe_present"))
            if primary not in output["logits"] or primary not in teacher_output["logits"]:
                raise ValueError(f"distillation teacher/student are missing primary head {primary!r}")
            temperature = float(distillation.get("temperature", 2.0))
            alpha_supervised = float(distillation.get("alpha_supervised", 1.0))
            alpha_distill = float(distillation.get("alpha_distill", 1.0))
            knowledge = knowledge_distillation_loss(
                output["logits"][primary],
                teacher_output["logits"][primary],
                temperature=temperature,
            )
            supervised = loss
            loss = alpha_supervised * supervised + alpha_distill * knowledge
            metrics["distillation.gt_loss"] = float(supervised.detach().cpu())
            metrics["distillation.kd_loss"] = float(knowledge.detach().cpu())
        metrics["total_loss"] = float(loss.detach().cpu())
        return loss, metrics

    def prognosis(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        from source.components.roi.masks import apply_counterfactual
        from source.components.targets import masked_multitask_loss

        model_batch = dict(batch)
        if "volume" in model_batch:
            model_batch["volume"] = apply_counterfactual(
                model_batch["volume"], model_batch["masks"], str(task.get("input_counterfactual") or ""),
                matched_region=str(task.get("matched_region", "pa")),
                masking_policy=task.get("masking_policy", "local_mean"),
                seed=counterfactual_seed,
                patient_ids=batch.get("patient_id"),
                study_ids=batch.get("study_id"),
            )
        output = model(model_batch)
        target_logits = output.get("target_logits")
        if target_logits:
            labels = {
                name: batch["labels"][:, label_columns.index(name)]
                for name in target_logits if name in label_columns
            }
            valid = {
                name: batch["label_valid"][:, label_columns.index(name)]
                for name in target_logits if name in label_columns
            }
            target_specs = task.get("targets") or {str(task.get("primary_target", "mortality_30d")): 1}
            loss, _ = masked_multitask_loss(
                target_logits, labels, valid, target_specs,
                task.get("loss_weights"), allow_empty=False,
            )
        else:
            index = int(task.get("primary_label_index", 0))
            loss = mortality_loss(
                output["logits"], batch["labels"][:, index], batch["label_valid"][:, index]
            )
        concept_logits = output.get("concept_logits")
        if concept_logits:
            from source.components.targets import masked_multitask_loss

            columns = output["concept_target_columns"]
            concept_labels = {
                name: batch["labels"][:, label_columns.index(column)]
                for name, column in columns.items()
                if column in label_columns
            }
            concept_valid = {
                name: batch["label_valid"][:, label_columns.index(column)]
                for name, column in columns.items()
                if column in label_columns
            }
            concept_loss, _ = masked_multitask_loss(
                concept_logits, concept_labels, concept_valid, output["concept_target_specs"],
                task.get("concept_loss_weights"), allow_empty=True,
            )
            loss = loss + float(task.get("concept_loss_weight", 0.2)) * concept_loss
        return loss

    def contour(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        return contour_loss(model(batch["volume"]), batch["masks"]["target"])

    callbacks = {"diagnosis": diagnosis, "prognosis": prognosis, "contour": contour}
    if stage not in callbacks:
        raise ValueError(f"no task loss for stage={stage}")
    callback = callbacks[stage]
    callback.teacher = teacher
    return callback
