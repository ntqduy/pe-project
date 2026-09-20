from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


METRIC_ORDER = (
    "auroc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "brier",
    "calibration_intercept",
    "calibration_slope",
)


def epoch_directory(run_dir: Path, epochs_run: int) -> Path:
    return run_dir / f"epoch_{int(epochs_run)}"


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    if not fields:
        fields = ["record_type"]
    preferred = [
        "record_type",
        "split",
        "cohort",
        "target",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "valid_replicates",
        "threshold",
        "threshold_source",
        "parameter_scope",
        "parameter",
        "parameter_value",
    ]
    fields = [field for field in preferred if field in fields] + [
        field for field in fields if field not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _flatten_parameters(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Flatten nested config values into stable CSV parameter rows."""
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_parameters(value[key], child))
        return rows
    if isinstance(value, (list, tuple)):
        return [{"parameter": prefix, "parameter_value": json.dumps(value, default=str)}]
    return [{"parameter": prefix, "parameter_value": _csv_value(value)}]


def _plot_training_curves(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for training-curve PDF output") from exc

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train loss")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    auc_columns = [("train_auroc", "train AUROC"), ("val_auroc", "val AUROC")]
    target_columns = sorted(
        key.removeprefix("train_").removesuffix("_auroc")
        for key in history[0]
        if key.startswith("train_") and key.endswith("_auroc") and key != "train_auroc"
    )
    if target_columns:
        auc_columns = []
        for target in target_columns:
            auc_columns.extend(
                ((f"train_{target}_auroc", f"train {target}"), (f"val_{target}_auroc", f"val {target}"))
            )
    plotted = False
    for column, label in auc_columns:
        values = [row.get(column) for row in history]
        if any(value not in (None, "", "nan") for value in values):
            axes[1].plot(epochs, [float(value) for value in values], label=label)
            plotted = True
    axes[1].set_title("Train / validation AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    if plotted:
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "AUC history unavailable", ha="center", va="center")
    figure.savefig(path, format="pdf")
    plt.close(figure)


def write_training_artifacts(
    run_dir: Path,
    training_result: Mapping[str, Any],
    *,
    lineage: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    extra_parameters: Mapping[str, Any] | None = None,
) -> Path:
    """Materialize the user-facing per-epoch artifact bundle.

    The historical root files remain in place for compatibility.  This bundle is a
    reproducible snapshot, with the validation-selected and final checkpoints side by side.
    """
    history = [dict(row) for row in training_result.get("history", [])]
    epochs_run = int(training_result.get("epochs_run") or (history[-1]["epoch"] if history else 0))
    if epochs_run < 1:
        raise ValueError("cannot write epoch artifacts without at least one completed epoch")
    destination = epoch_directory(run_dir, epochs_run)
    checkpoint_dir = destination / "checkpoint"
    preview_dir = destination / "preview"
    plots_dir = destination / "plots"
    for directory in (checkpoint_dir, preview_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for name in ("best.ckpt", "last.ckpt"):
        source = run_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"expected checkpoint was not created: {source}")
        shutil.copy2(source, checkpoint_dir / name)

    run_log = run_dir / "logs" / "run.log"
    if run_log.is_file():
        shutil.copy2(run_log, destination / "logs.txt")
    else:
        (destination / "logs.txt").write_text("run.log was not created\n", encoding="utf-8")

    result_rows = [{"record_type": "epoch", "split": "train_validation", **row} for row in history]
    if config is not None:
        result_rows.extend(
            {
                "record_type": "parameter",
                "parameter_scope": "config",
                **row,
            }
            for row in _flatten_parameters(config)
        )
    if extra_parameters:
        result_rows.extend(
            {
                "record_type": "parameter",
                "parameter_scope": "runtime",
                **row,
            }
            for row in _flatten_parameters(extra_parameters)
        )
    _write_rows(destination / "result.csv", result_rows)
    _plot_training_curves(plots_dir / "training_curves.pdf", history)
    metadata = {
        "epochs_configured": int(training_result.get("epochs", epochs_run)),
        "epochs_run": epochs_run,
        "early_stopping_patience": training_result.get("early_stopping_patience"),
        "stopped_early": bool(training_result.get("stopped_early", False)),
        "checkpoint": {"best": "checkpoint/best.ckpt", "last": "checkpoint/last.ckpt"},
        "preview_split": "validation",
        "split_policy": "read-only official train/validation/test manifest; no repartition",
        "lineage": dict(lineage or {}),
    }
    (destination / "artifacts.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return destination


def append_evaluation_result_csv(
    run_dir: Path,
    result_evaluation: Mapping[str, Any],
    *,
    epochs_run: int | None = None,
) -> Path | None:
    candidates = []
    if epochs_run:
        candidates.append(epoch_directory(run_dir, int(epochs_run)))
    candidates.extend(sorted(run_dir.glob("epoch_*"), reverse=True))
    destination = next((item for item in candidates if item.is_dir()), None)
    if destination is None:
        return None
    rows: list[dict[str, Any]] = []
    existing = destination / "result.csv"
    if existing.is_file():
        with existing.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row.get("record_type")
        not in {"final_metric", "evaluation_summary", "evaluation_parameter"}
    ]
    targets = result_evaluation.get("targets") or {
        result_evaluation.get("primary_target", ""): result_evaluation
    }
    cohort = result_evaluation.get("cohort", "")
    for target, target_result in targets.items():
        target_result = dict(target_result or {})
        metrics = target_result.get("metrics") or {}
        ordered_metrics = sorted(
            metrics.items(),
            key=lambda item: (
                METRIC_ORDER.index(str(item[0]))
                if str(item[0]) in METRIC_ORDER
                else len(METRIC_ORDER),
                str(item[0]),
            ),
        )
        for name, values in ordered_metrics:
            item = {
                "record_type": "final_metric",
                "split": "test",
                "cohort": cohort,
                "target": target,
                "metric": name,
            }
            if isinstance(values, Mapping):
                item.update(values)
            else:
                item["value"] = values
            rows.append(item)
        for name in ("threshold", "evaluated_patients", "validation_patients"):
            if name in target_result:
                rows.append(
                    {
                        "record_type": "evaluation_summary",
                        "split": "test",
                        "cohort": cohort,
                        "target": target,
                        "metric": name,
                        "value": target_result[name],
                    }
                )
        for name, value in {
            "bootstrap.unit": (target_result.get("bootstrap") or {}).get("unit"),
            "bootstrap.samples": (target_result.get("bootstrap") or {}).get("samples"),
            "bootstrap.confidence": (target_result.get("bootstrap") or {}).get("confidence"),
            "threshold_source": target_result.get("threshold_source"),
            "status": target_result.get("status"),
            "unavailable_reason": target_result.get("unavailable_reason"),
        }.items():
            if value not in (None, ""):
                rows.append(
                    {
                        "record_type": "evaluation_parameter",
                        "split": "test",
                        "cohort": cohort,
                        "target": target,
                        "parameter": name,
                        "parameter_value": value,
                    }
                )
    _write_rows(existing, rows)
    return existing


def refresh_epoch_log(run_dir: Path, *, epochs_run: int | None = None) -> Path | None:
    candidates = []
    if epochs_run:
        candidates.append(epoch_directory(run_dir, int(epochs_run)))
    candidates.extend(sorted(run_dir.glob("epoch_*"), reverse=True))
    destination = next((item for item in candidates if item.is_dir()), None)
    source = run_dir / "logs" / "run.log"
    if destination is None or not source.is_file():
        return None
    target = destination / "logs.txt"
    shutil.copy2(source, target)
    return target


def append_epoch_parameters(
    run_dir: Path,
    parameters: Mapping[str, Any],
    *,
    scope: str = "runtime",
    epochs_run: int | None = None,
) -> Path | None:
    candidates = []
    if epochs_run:
        candidates.append(epoch_directory(run_dir, int(epochs_run)))
    candidates.extend(sorted(run_dir.glob("epoch_*"), reverse=True))
    destination = next((item for item in candidates if item.is_dir()), None)
    if destination is None:
        return None
    result_path = destination / "result.csv"
    rows: list[dict[str, Any]] = []
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    flattened = _flatten_parameters(parameters)
    existing_keys = {
        (str(row.get("parameter_scope", "")), str(row.get("parameter", "")))
        for row in rows
        if row.get("record_type") == "parameter"
    }
    rows.extend(
        {
            "record_type": "parameter",
            "parameter_scope": scope,
            **row,
        }
        for row in flattened
        if (scope, str(row.get("parameter", ""))) not in existing_keys
    )
    _write_rows(result_path, rows)
    return result_path


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _primary_logit(output: Mapping[str, Any], stage: str, target: str) -> Tensor:
    if stage == "diagnosis":
        return output["logits"][target].reshape(-1)[0]
    target_logits = output.get("target_logits")
    if target_logits and target in target_logits:
        return target_logits[target].reshape(-1)[0]
    return output["logits"].reshape(-1)[0]


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _write_heatmap(image: Tensor, heatmap: Tensor, destination: Path, title: str) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    array = image.detach().float().cpu().numpy()
    overlay = heatmap.detach().float().cpu().numpy()
    lower, upper = np.percentile(array, [1, 99])
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)) or upper <= lower:
        lower, upper = float(array.min()), float(array.max() or 1.0)
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.imshow(array.T, cmap="gray", origin="lower", vmin=lower, vmax=upper)
    axis.imshow(overlay.T, cmap="turbo", origin="lower", alpha=0.48, vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.axis("off")
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def write_backbone_previews(
    model: nn.Module,
    dataset: Any,
    config: Mapping[str, Any],
    device: torch.device,
    destination: Path,
    *,
    maximum_patients: int = 5,
) -> dict[str, Any]:
    """Write two middle-slice diagnostics per validation patient.

    ``feature_activation`` shows mean absolute backbone activation; ``gradcam`` is
    target-specific and is computed from the selected primary logit.  This is qualitative
    validation-only output and never changes the official split or task metrics.
    """
    destination.mkdir(parents=True, exist_ok=True)
    stage = str((config.get("experiment") or {}).get("stage") or "")
    if stage in {"ablation", "roi_student"}:
        stage = str((config.get("task") or {}).get("base_stage") or stage)
    target = str((config.get("task") or {}).get("primary_target") or "pe_present")
    underlying = _unwrap(model)
    encoder = getattr(underlying, "image_encoder", None)
    if encoder is None:
        note = "No image backbone is present; preview skipped (clinical-only model).\n"
        (destination / "README.txt").write_text(note, encoding="utf-8")
        return {"status": "skipped", "reason": "no_image_backbone", "patients": 0}

    written = 0
    errors: list[str] = []
    hook_state: dict[str, Tensor] = {}

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        feature_map = getattr(output, "feature_map", None)
        if feature_map is None and isinstance(output, Mapping):
            feature_map = output.get("feature_map")
        if feature_map is None:
            raise RuntimeError("image encoder output did not expose feature_map")
        hook_state["feature_map"] = feature_map
        feature_map.retain_grad()

    handle = encoder.register_forward_hook(capture)
    try:
        model.eval()
        for index in range(min(int(maximum_patients), len(dataset))):
            item = dataset[index]
            patient = str(item.get("patient_id", f"patient_{index:03d}"))
            study = str(item.get("study_id", f"study_{index:03d}"))
            try:
                batch = dict(item)
                for key in ("volume", "labels", "ehr", "spesi", "ehr_available", "spesi_available"):
                    if isinstance(batch.get(key), Tensor):
                        batch[key] = batch[key].unsqueeze(0)
                if isinstance(batch.get("masks"), Mapping):
                    batch["masks"] = {
                        name: value.unsqueeze(0) if isinstance(value, Tensor) else value
                        for name, value in batch["masks"].items()
                    }
                batch = _move_preview_batch(batch, device)
                volume = batch["volume"].detach().float().requires_grad_(True)
                batch["volume"] = volume
                model.zero_grad(set_to_none=True)
                hook_state.clear()
                if stage == "diagnosis":
                    output = model(volume, batch["masks"])
                elif stage == "prognosis":
                    output = model(batch)
                else:
                    continue
                probability = float(torch.sigmoid(_primary_logit(output, stage, target)).detach().cpu())
                _primary_logit(output, stage, target).backward()
                feature_map = hook_state.get("feature_map")
                if feature_map is None or feature_map.grad is None:
                    raise RuntimeError("backbone feature map gradient was unavailable")
                activation = feature_map.detach().abs().mean(dim=1, keepdim=True)
                gradient = feature_map.grad.detach()
                weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
                gradcam = (weights * feature_map.detach()).sum(dim=1, keepdim=True).relu()
                size = tuple(int(value) for value in volume.shape[-3:])
                activation = F.interpolate(activation, size=size, mode="trilinear", align_corners=False)[0, 0]
                gradcam = F.interpolate(gradcam, size=size, mode="trilinear", align_corners=False)[0, 0]
                activation = activation / activation.amax().clamp_min(1e-8)
                gradcam = gradcam / gradcam.amax().clamp_min(1e-8)
                middle = int(volume.shape[-3] // 2)
                image = volume[0, 0, middle]
                prefix = f"{written + 1:02d}_{_safe_name(patient)}_{_safe_name(study)}"
                _write_heatmap(
                    image,
                    activation[:, :, middle],
                    destination / f"{prefix}_feature_activation.png",
                    f"feature activation | p={probability:.3f} | z={middle}",
                )
                _write_heatmap(
                    image,
                    gradcam[:, :, middle],
                    destination / f"{prefix}_gradcam.png",
                    f"target Grad-CAM ({target}) | p={probability:.3f} | z={middle}",
                )
                written += 1
            except Exception as exc:  # preview is diagnostic; one bad case must not erase metrics
                errors.append(f"{patient}/{study}: {type(exc).__name__}: {exc}")
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    readme = (
        "Preview source split: validation (first five manifest rows).\n"
        "feature_activation: mean absolute backbone feature activation.\n"
        "gradcam: target-specific gradient-weighted feature activation.\n"
        "These images are qualitative diagnostics only; they are not task metrics.\n"
    )
    if errors:
        readme += "\nErrors:\n" + "\n".join(errors) + "\n"
    (destination / "README.txt").write_text(readme, encoding="utf-8")
    return {"status": "completed", "patients": written, "errors": errors}


def _move_preview_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    def move(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    return move(dict(batch))


__all__ = [
    "append_evaluation_result_csv",
    "append_epoch_parameters",
    "epoch_directory",
    "refresh_epoch_log",
    "write_backbone_previews",
    "write_training_artifacts",
]
