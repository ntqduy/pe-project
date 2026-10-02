from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from source.components.targets import normalize_target_spec
from source.tasks.diagnosis.losses import diagnosis_loss, organ_auxiliary_loss
from source.tasks.prognosis.losses import mortality_loss


def diagnosis_evaluation_targets(
    task: Mapping[str, Any], label_columns: Sequence[str], primary: str
) -> list[str]:
    """Diagnosis targets scored by evaluation and the epoch AUROC pass, primary first.

    Besides the primary target these are the other native binary heads (``task.targets``
    entries with a ``data.label_columns`` column, e.g. multitask pe_acute/pe_subsegmental).
    They only add rows: the primary target's metrics, threshold and the checkpoint selection
    are unchanged. Silver and multiclass heads have no native label column and are skipped.
    """
    columns = {str(column) for column in label_columns}
    silver = {str(name) for name in task.get("silver_targets") or ()}
    configured = task.get("targets") or {}
    if not isinstance(configured, Mapping):
        configured = {str(name): 1 for name in configured}
    targets = [str(primary)]
    for name, spec in configured.items():
        name = str(name)
        if name in targets or name not in columns or name in silver:
            continue
        try:
            binary = normalize_target_spec(spec).kind == "binary"
        except (TypeError, ValueError):
            binary = False
        if binary:
            targets.append(name)
    return targets


def task_loss_step(config: Mapping[str, Any]):
    stage = str((config.get("experiment") or {}).get("stage"))
    data = dict(config.get("data") or {})
    task = dict(config.get("task") or {})
    if stage == "ablation":
        stage = str(task.get("base_stage") or "")
    label_columns = tuple(data.get("label_columns") or ())
    counterfactual_seed = int(config.get("seed", 42))

    def diagnosis(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        from source.components.roi.masks import apply_counterfactual

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
        return loss

    callbacks = {"diagnosis": diagnosis, "prognosis": prognosis}
    if stage not in callbacks:
        raise ValueError(f"no task loss for stage={stage}")
    return callbacks[stage]
