from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from source.components.targets import masked_multitask_loss, normalize_target_spec
from source.tasks.diagnosis.losses import diagnosis_loss


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
    """Loss of one batch for the baseline classifier (source/model/classifier.py).

    diagnosis: summed BCE / CE over the configured targets that have a label column;
    prognosis: masked multitask loss over the configured binary outcomes (censored rows masked).
    """
    stage = str((config.get("experiment") or {}).get("stage"))
    data = dict(config.get("data") or {})
    task = dict(config.get("task") or {})
    label_columns = tuple(data.get("label_columns") or ())

    def diagnosis(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        output = model(batch["volume"], batch["masks"])
        labels = {name: batch["labels"][:, index] for index, name in enumerate(label_columns)}
        valid = {name: batch["label_valid"][:, index] for index, name in enumerate(label_columns)}
        loss, main_report = diagnosis_loss(output["logits"], labels, valid, task.get("loss_weights"))
        metrics: dict[str, float] = {f"main.{name}.loss": value for name, value in main_report.items()}
        for name in output["logits"]:
            if name in valid:
                metrics[f"main.{name}.valid_count"] = float(valid[name].sum().detach().cpu())
        metrics["total_loss"] = float(loss.detach().cpu())
        return loss, metrics

    def prognosis(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        output = model(dict(batch))
        target_logits = output["target_logits"]
        labels = {
            name: batch["labels"][:, label_columns.index(name)]
            for name in target_logits if name in label_columns
        }
        valid = {
            name: batch["label_valid"][:, label_columns.index(name)]
            for name in target_logits if name in label_columns
        }
        target_specs = task.get("targets") or {str(task.get("primary_target")): 1}
        loss, _ = masked_multitask_loss(
            target_logits, labels, valid, target_specs, task.get("loss_weights"), allow_empty=False,
        )
        return loss

    callbacks = {"diagnosis": diagnosis, "prognosis": prognosis}
    if stage not in callbacks:
        raise ValueError(f"no task loss for stage={stage}")
    return callbacks[stage]
