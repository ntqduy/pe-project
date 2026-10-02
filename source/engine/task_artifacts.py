from __future__ import annotations

import copy
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from source.imaging.cam_preview import CAM_MEANING, CAM_METHODS, cam_from_target_layer

# Files of earlier artifact contracts that must not linger beside the current bundle.
LEGACY_EPOCH_FILES = ("training_curves.pdf", "artifacts.json")
LEGACY_PREVIEW_FILES = ("README.txt", "summary.json")


def latest_epoch_directory(run_dir: Path) -> Path | None:
    """The epoch bundle with the highest epoch number (numeric, not lexicographic)."""
    candidates = []
    for path in run_dir.glob("epoch_*"):
        suffix = path.name.removeprefix("epoch_")
        if path.is_dir() and suffix.isdigit():
            candidates.append((int(suffix), path))
    return max(candidates)[1] if candidates else None


def best_epoch(history: Sequence[Mapping[str, Any]]) -> int | None:
    """Epoch saved as best.ckpt: the first epoch with the highest selection metric."""
    best_value = float("-inf")
    selected: int | None = None
    for row in history:
        value = _number(row.get("primary_val_metric"))
        if value is not None and value > best_value:
            best_value, selected = value, int(row["epoch"])
    return selected


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plot_training_curves(
    path: Path,
    history: Sequence[Mapping[str, Any]],
    *,
    selected_epoch: int | None = None,
    title: str = "",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for training-curve output") from exc

    epochs = [int(row["epoch"]) for row in history]

    def series(column: str) -> list[float]:
        return [
            value if (value := _number(row.get(column))) is not None else float("nan")
            for row in history
        ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    # One rule in both panels: validation = solid line + filled marker, train = dashed line +
    # hollow marker. Markers keep a one-epoch smoke run visible; lines alone draw nothing for
    # one point.
    split_style = {
        "train": {"color": "C0", "linestyle": "--", "marker": "o", "markerfacecolor": "white"},
        "validation": {"color": "C1", "linestyle": "-", "marker": "o"},
    }
    axes[0].plot(epochs, series("train_loss"), label="train loss", **split_style["train"])
    axes[0].plot(epochs, series("val_loss"), label="val loss", **split_style["validation"])
    axes[0].set_title("Loss")
    axes[0].set_ylabel("loss")

    # Per-target keys keep the configured order; train_auroc/val_auroc duplicate the primary.
    targets = [
        key.removeprefix("train_").removesuffix("_auroc")
        for key in (history[0] if history else {})
        if key.startswith("train_") and key.endswith("_auroc") and key != "train_auroc"
    ]
    handles, labels = [], []
    plotted = False
    if len(targets) > 1:
        # Several targets: colour names the target, line style names the split.
        palette = plt.get_cmap("tab10")
        for index, target in enumerate(targets):
            colour = palette(index % 10)
            for split, prefix in (("validation", "val"), ("train", "train")):
                values = series(f"{prefix}_{target}_auroc")
                if any(math.isfinite(value) for value in values):
                    faded = {"alpha": 0.6} if split == "train" else {}
                    axes[1].plot(epochs, values, **{**split_style[split], "color": colour, **faded})
                    plotted = True
            handles.append(Line2D([0], [0], color=colour, lw=2))
            labels.append(target)
        for split in ("validation", "train"):
            handles.append(Line2D([0], [0], **{**split_style[split], "color": "0.3"}))
        labels += ["validation (solid, filled)", "train (dashed, hollow)"]
        axes[1].set_title("AUROC per target (solid = validation, dashed = train)")
    else:
        # One target: the same train/validation colours as the loss panel.
        target = targets[0] if targets else None
        infix = f"{target}_" if target else ""
        for split, prefix in (("train", "train"), ("validation", "val")):
            values = series(f"{prefix}_{infix}auroc")
            if any(math.isfinite(value) for value in values):
                (line,) = axes[1].plot(epochs, values, **split_style[split])
                handles.append(line)
                labels.append(f"{prefix} AUROC")
                plotted = True
        axes[1].set_title(f"AUROC ({target})" if target else "AUROC")
    axes[1].set_ylabel("AUROC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].axhline(0.5, color="0.6", linewidth=0.8, linestyle=":")
    if not plotted:
        axes[1].text(0.5, 0.5, "AUROC undefined (a split has one class)", ha="center", va="center")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(epochs) == 1:
            axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
        if selected_epoch is not None:
            axis.axvline(selected_epoch, color="0.4", linestyle=":", linewidth=1.2)
    if selected_epoch is not None:
        axes[0].plot([], [], color="0.4", linestyle=":", label=f"best.ckpt (epoch {selected_epoch})")
        handles.append(Line2D([0], [0], color="0.4", linestyle=":"))
        labels.append(f"best.ckpt (epoch {selected_epoch})")
    axes[0].legend(fontsize=8)
    if handles and len(targets) > 1:
        axes[1].legend(handles, labels, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    elif handles:
        axes[1].legend(handles, labels, fontsize=8)
    if title:
        figure.suptitle(title)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_training_artifacts(
    run_dir: Path,
    training_result: Mapping[str, Any],
    *,
    lineage: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Materialize the user-facing artifact bundle inside the run's ``epoch_<E>/`` folder.

    ``run_dir`` is the bundle itself (``<id>/epoch_<training.epochs>``, see
    ``source.engine.experiment.run_output_id``). It receives checkpoint/{best,last}.ckpt,
    logs.txt, history.csv, training_curves.png and preview/. result.csv and predictions.csv are added by
    tools/tasks/evaluate.py, which is the only place task metrics are computed.
    """
    history = [dict(row) for row in training_result.get("history", [])]
    epochs_run = int(training_result.get("epochs_run") or (history[-1]["epoch"] if history else 0))
    if epochs_run < 1:
        raise ValueError("cannot write epoch artifacts without at least one completed epoch")
    epochs_configured = int(training_result.get("epochs") or epochs_run)
    destination = run_dir
    checkpoint_dir = destination / "checkpoint"
    preview_dir = destination / "preview"
    for directory in (checkpoint_dir, preview_dir):
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

    history_csv = run_dir / "logs" / "history.csv"
    if not history_csv.is_file():
        raise FileNotFoundError(f"expected training history was not created: {history_csv}")
    shutil.copy2(history_csv, destination / "history.csv")

    experiment_id = str(((config or {}).get("experiment") or {}).get("id") or run_dir.name)
    # The folder is named after the budget; say so when early stopping ended the run sooner.
    epochs_label = (
        f"{epochs_run} epoch(s)"
        if epochs_run >= epochs_configured
        else f"{epochs_run}/{epochs_configured} epoch(s), stopped early"
    )
    _plot_training_curves(
        destination / "training_curves.png",
        history,
        selected_epoch=best_epoch(history),
        title=f"{experiment_id} | {epochs_label}",
    )
    for name in LEGACY_EPOCH_FILES:
        (destination / name).unlink(missing_ok=True)
    return destination


def cleanup_task_run(run_dir: Path, *, keep: Sequence[Path] = ()) -> None:
    """Remove compatibility duplicates after a task has produced its epoch bundle.

    ``keep`` names files the caller has just written at the run root; they are current
    outputs, not the legacy root copies matched below. bootstrap_metrics.parquet and
    reporting_checklist.json are never removed here: a full-test evaluation writes them at
    the run root in every layout, so they are not legacy copies.
    """
    kept = {Path(path).resolve() for path in keep}
    # A root checkpoint is only a duplicate once the bundle holds its copy; in an older layout
    # without checkpoint/ it is the only copy and must survive.
    bundles = (run_dir / "checkpoint", *(path / "checkpoint" for path in run_dir.glob("epoch_*")))
    for name in ("best.ckpt", "last.ckpt"):
        path = run_dir / name
        if path.is_file() and any((bundle / name).is_file() for bundle in bundles):
            path.unlink()
    has_bundle_checkpoint = any((bundle / "best.ckpt").is_file() for bundle in bundles)
    if not has_bundle_checkpoint:
        kept.update((run_dir / name).resolve() for name in ("best.ckpt.metadata.json", "last.ckpt.metadata.json"))
    for name in (
        "best.ckpt.metadata.json",
        "last.ckpt.metadata.json",
        "config.yaml",
        "metrics.json",
        "lineage.json",
        "environment.json",
    ):
        path = run_dir / name
        if path.is_file() and path.resolve() not in kept:
            path.unlink()
    for name in ("logs", "checkpoints", "figures", "qc"):
        path = run_dir / name
        # An old checkpoints/ folder may hold the only weights of a pre-bundle run.
        if name == "checkpoints" and not has_bundle_checkpoint:
            continue
        if path.is_dir():
            shutil.rmtree(path)
    # Predictions now live in epoch_<N>/predictions.csv; root copies are legacy.
    for pattern in ("predictions.parquet", "calibration_curve*.parquet"):
        for path in run_dir.glob(pattern):
            if path.is_file() and path.resolve() not in kept:
                path.unlink()
    # run_dir is normally the epoch_<E> bundle itself; older run folders hold epoch_<N>/.
    for epoch_dir in (run_dir, *run_dir.glob("epoch_*")):
        if not epoch_dir.is_dir():
            continue
        plots_dir = epoch_dir / "plots"
        if plots_dir.is_dir():
            shutil.rmtree(plots_dir)
        for name in LEGACY_EPOCH_FILES:
            path = epoch_dir / name
            # A PDF curve is legacy only once the PNG replacement exists.
            if name == "training_curves.pdf" and not (epoch_dir / "training_curves.png").is_file():
                continue
            if path.is_file():
                path.unlink()
        for name in LEGACY_PREVIEW_FILES:
            (epoch_dir / "preview" / name).unlink(missing_ok=True)


def refresh_epoch_log(run_dir: Path) -> Path | None:
    """Copy the training run.log into the bundle's logs.txt (training phase only).

    Evaluation appends to logs.txt directly; calling this after evaluation would erase it.
    """
    source = run_dir / "logs" / "run.log"
    if not source.is_file():
        return None
    target = run_dir / "logs.txt"
    shutil.copy2(source, target)
    return target


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


def _base_dataset(dataset: Any) -> Any | None:
    """The CTPA dataset under simple wrapper datasets (it owns rows/_resolve/transform)."""
    current = dataset
    for _ in range(4):
        if current is None:
            return None
        if getattr(current, "rows", None) is not None and callable(getattr(current, "_resolve", None)):
            return current
        current = getattr(current, "base", None) or getattr(current, "dataset", None)
    return None


POOLED_FEATURE_COLUMN = "pooled_path"    # manifests/ct_fm: pooled [513, 1, 1, 1] copy
GRID_FEATURE_COLUMN = "image_path"       # manifests/ct_fm: the cached [513, d, h, w] grid


def _grid_view(dataset: Any) -> Any:
    """``dataset`` reading the CT-FM grid when it serves pooled features, else unchanged.

    A pooled input predicts exactly what its grid does but has no spatial map to explain, so
    previews of a ``data.file_column: pooled_path`` run use the grid of the same studies;
    their forward pass is still checked against the pooled run's probabilities.
    """
    base = _base_dataset(dataset)
    if base is None or getattr(base, "image_column", None) != POOLED_FEATURE_COLUMN:
        return dataset
    if not base.rows or GRID_FEATURE_COLUMN not in base.rows[0]:
        return dataset
    view = copy.copy(base)
    view.image_column = GRID_FEATURE_COLUMN
    if dataset is base:
        return view
    wrapper = copy.copy(dataset)
    for attribute in ("base", "dataset"):
        if getattr(wrapper, attribute, None) is base:
            setattr(wrapper, attribute, view)
            return wrapper
    return dataset


def _dataset_image_path(dataset: Any, index: int) -> Path | None:
    """Resolved volume path of ``dataset[index]`` (follows simple wrapper datasets)."""
    base = _base_dataset(dataset)
    column = getattr(base, "image_column", None)
    if base is None or not column:
        return None
    try:
        return Path(base._resolve(base.rows[index][column]))
    except (IndexError, KeyError, TypeError):
        return None


def _volume_geometry(
    metadata: Mapping[str, Any] | None,
) -> tuple[str | None, tuple[float, float, float] | None]:
    """Array orientation and voxel spacing recorded by the preprocessing sidecar.

    ``orientation`` (e.g. "SPL" for the CT-FM canvas) and ``output_spacing_mm`` are in
    array-axis order. Either is None when the sidecar does not record it; the preview then
    leaves the array unrotated and labels the orientation N/A instead of assuming one.
    """
    if not metadata:
        return None, None
    orientation: str | None = None
    code = str(metadata.get("orientation") or "").upper()
    axes = ({"L", "R"}, {"A", "P"}, {"S", "I"})
    if len(code) == 3 and all(len(set(code) & pair) == 1 for pair in axes):
        orientation = code
    spacing: tuple[float, float, float] | None = None
    try:
        values = tuple(float(value) for value in metadata.get("output_spacing_mm"))
        if len(values) == 3 and all(math.isfinite(value) and value > 0 for value in values):
            spacing = (values[0], values[1], values[2])
    except (TypeError, ValueError):
        spacing = None
    return orientation, spacing


def _display_ct_for_features(image_path: Path | None) -> tuple[Any, dict[str, Any]]:
    """CT canvas a cached-feature file was computed from, rebuilt from its raw NIfTI.

    Returns the canvas and the metadata of this rebuild, so the caller can check it against
    the cache sidecar (same shape, affine and bounds means the same canvas CT-FM received).
    """
    from source.data.build.volumes import PreprocessingSpec, preprocess_volume_with_metadata
    from source.imaging.grid import read_sidecar

    sidecar = read_sidecar(image_path) if image_path is not None else None
    if not sidecar or not sidecar.get("spec") or not sidecar.get("raw_image_path"):
        raise RuntimeError(f"cached features have no preprocessing sidecar to rebuild the CT: {image_path}")
    payload = dict(sidecar["spec"])
    payload.pop("extra", None)
    payload.pop("implementation", None)
    return preprocess_volume_with_metadata(sidecar["raw_image_path"], PreprocessingSpec.from_mapping(payload))


def _to_spl(
    arrays: Sequence[Any],
    orientation: str | None,
    spacing: tuple[float, float, float] | None,
) -> tuple[list[Any], tuple[float, float, float] | None]:
    """Reorient 3D arrays to S,P,L (axial slice = array[z] with anterior in row 0).

    With an unknown orientation the arrays and spacing are returned unchanged.
    """
    import nibabel as nib
    import numpy as np

    if orientation is None:
        return [np.asarray(array) for array in arrays], spacing
    transform = nib.orientations.ornt_transform(
        nib.orientations.axcodes2ornt(tuple(orientation)),
        nib.orientations.axcodes2ornt(("S", "P", "L")),
    )
    oriented = [np.asarray(nib.orientations.apply_orientation(np.asarray(array), transform)) for array in arrays]
    if spacing is None:
        return oriented, None
    target = [0.0, 0.0, 0.0]
    for source_axis, (target_axis, _) in enumerate(transform):
        target[int(target_axis)] = float(spacing[source_axis])
    return oriented, (target[0], target[1], target[2])


# Preview forward vs evaluate probability; larger gaps mean the preview explains another model.
PROBABILITY_TOLERANCE = 1e-4


def _pooling_from_gradient(gradient: Tensor | None) -> dict[str, Any]:
    """How the head pools the target layer, read off ∂y/∂A instead of assumed.

    Plain global average pooling gives every cell the same gradient per channel; weighted
    (body coverage, ROI mask) pooling does not.
    """
    if gradient is None:
        return {"uniform": False, "spread": float("nan"), "text": "N/A (không có gradient)"}
    values = gradient.detach()[0].double().flatten(1)
    scale = float(values.abs().max())
    spread = float((values.amax(dim=1) - values.amin(dim=1)).max()) / scale if scale > 0 else 0.0
    uniform = spread < 1e-6
    text = (
        f"∂logit/∂A đồng nhất theo không gian (lệch tương đối tối đa {spread:.1e}) → head dùng "
        "trung bình đều trên toàn lưới feature, kể cả ô padding của canvas; khi đó HiResCAM = "
        "Grad-CAM gốc"
        if uniform
        else f"∂logit/∂A thay đổi theo không gian (lệch tương đối {spread:.2g}) → pooling có trọng "
        "số hoặc theo ROI"
    )
    return {"uniform": uniform, "spread": spread, "text": text}


def _threshold_rule_text(rule: str | None) -> str:
    text = str(rule or "")
    if text.endswith("_on_validation"):
        method = text.removesuffix("_on_validation")
        detail = "Youden J = sensitivity + specificity − 1 cực đại" if method == "youden" else method
        return f"{text}: {detail} trên validation, chọn bởi evaluate; preview không chọn lại"
    if text == "default_0.5_validation_one_class":
        return f"{text}: 0.5 mặc định vì validation chỉ có một lớp"
    if text == "locked_internal_validation":
        return f"{text}: khóa từ validation nội bộ"
    return text or "N/A"


def _outcome(truth: int | None, predicted: int | None) -> str:
    if truth is None or predicted is None:
        return ""
    return ("TP" if truth else "FP") if predicted else ("FN" if truth else "TN")


def _metadata_mismatches(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    import numpy as np

    different = []
    for key in ("output_shape", "output_affine", "target_bounds", "source_bounds", "crop_box"):
        left, right = first.get(key), second.get(key)
        try:
            same = np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), atol=1e-4)
        except (TypeError, ValueError):
            same = left == right
        if not same:
            different.append(key)
    return different


def _describe_target_layer(encoder: nn.Module, hook_module: nn.Module, config: Mapping[str, Any], channels: int) -> str:
    backbone = str(getattr(encoder, "backbone_name", "") or type(encoder).__name__)
    if bool((config.get("model") or {}).get("cached_features")):
        return (
            f"CT-FM SegResEncoder layers[4].blocks — stage sâu nhất, {channels} kênh. Run này dùng "
            "feature cache: tensor đó chính là input của phần model được train (kênh 0–511 của "
            "file cache), nên ∂logit/∂A được lấy trực tiếp qua pooling → adapter → head; CT-FM "
            "đã đóng băng, không cần backprop vào nó"
        )
    return (
        f"output 5D sâu nhất của backbone {backbone} ({type(hook_module).__name__}, {channels} kênh), "
        "tầng cuối trước pooling toàn cục/ROI; gradient đi qua toàn bộ phần sau nó"
    )


def _technical_rows(
    *,
    experiment_id: str,
    checkpoint: Path | str | None,
    checkpoint_sha256: str | None,
    config: Mapping[str, Any],
    stage: str,
    encoder: nn.Module,
    hook_module: nn.Module,
    cam_method: str,
    input_shape: tuple[int, ...],
    display_shape: tuple[int, int, int],
    canvas_source: str,
    channels: int,
    grid_shape: tuple[int, int, int],
    spacing: tuple[float, float, float] | None,
    vmax: float,
    hu_known: bool,
    orientation_text: str,
    display_spacing: tuple[float, float, float] | None,
    mapping: Mapping[str, Any],
    raw_cam: Any,
    outside: float | None,
    padding: float | None,
    pooling_text: str,
    z_profile: Any,
    oriented: bool,
    reference: float | None,
    preview_probability: float,
    selection_text: str = "các study đầu của validation theo thứ tự manifest; không chọn theo đúng/sai",
) -> list[tuple[str, str]]:
    """(label, value) rows of the collapsible technical section of one preview."""
    from source.imaging.cam_preview import contribution_text, display_rows, format_number

    def shape(values: Sequence[int]) -> str:
        return " × ".join(str(int(value)) for value in values)

    cached = bool((config.get("model") or {}).get("cached_features"))
    cells = tuple(display_shape[axis] / grid_shape[axis] for axis in range(3))
    cell_text = " × ".join(format_number(value, 4) for value in cells) + " voxel"
    if spacing:
        millimetres = " × ".join(format_number(value * step, 4) for value, step in zip(cells, spacing))
        cell_text += f" = {millimetres} mm (thứ tự trục của mảng input)"
    if any(abs(value - round(value)) > 1e-6 for value in cells):
        cell_text += "; không chia hết — ánh xạ ô → voxel chỉ gần đúng"
    if mapping.get("available"):
        mapped = sum(entry.get("status") == "mapped" for entry in mapping.get("entries") or ())
        mapping_text = (
            f"{mapped}/{len(mapping['entries'])} slice đầu vào có dữ liệu quét, phần còn lại là "
            "padding; ánh xạ theo chuỗi canvas crop/pad → body crop → ndimage.zoom → đảo trục ghi trong sidecar; slice gốc dày "
            f"{format_number(mapping['original_spacing_mm'], 4)} mm (trục {mapping['axis_name']}); "
            "affine sidecar lệch chuỗi zoom thực tế tối đa "
            f"{format_number(mapping['affine_discrepancy_mm'], 3)} mm"
        )
    else:
        mapping_text = f"N/A — {mapping.get('reason')}"
    architecture = (
        "Model 3D: một forward trên cả volume. CT-FM chạy trên các patch 3D 24×128×128 không chồng "
        "lấn rồi ghép thành một feature map; không có bước gộp slice 2D"
        if cached
        else getattr(encoder, "architecture_note", "")
        or "Model 3D: một forward trên cả volume; không có bước gộp slice 2D"
    )
    spacing_text = (
        " × ".join(format_number(value, 4) for value in display_spacing) + " mm"
        if display_spacing else "N/A (pixel vuông)"
    )
    probability_check = (
        "N/A (không có predictions của evaluate)"
        if reference is None
        else f"evaluate {float(reference):.6f} vs forward preview {preview_probability:.6f}, "
        f"|Δ| = {abs(preview_probability - float(reference)):.2e}"
    )
    padding_text = (
        "N/A (không có canvas crop/pad trong sidecar)"
        if padding is None
        else f"{100 * padding:.1f}% khối CAM dương (lưới hiển thị) nằm ngoài hộp dữ liệu quét "
        "target_bounds của canvas"
    )
    profile_text = (
        "N/A"
        if z_profile is None
        else " · ".join(f"{100 * float(value):.1f}%" for value in z_profile)
        + (" (ô z từ dưới lên" if oriented else " (theo trục 0 của mảng, hướng không xác định")
        + ("; mỗi ô feature tương ứng nhiều lát CT canvas)" if cached else
           "; mỗi ô tương ứng nhiều lát CT đầu vào)")
    )
    outside_text = (
        "N/A"
        if outside is None
        else f"{100 * outside:.1f}% khối CAM dương nằm ngoài mask cơ thể thô (HU > −500, thành phần "
        "lớn nhất, lấp lỗ từng slice); mask chỉ dùng cho con số này, không che heatmap"
    )
    return [
        ("Experiment", experiment_id),
        ("Checkpoint", str(checkpoint) if checkpoint else "N/A"),
        ("Checkpoint SHA-256", checkpoint_sha256 or "N/A"),
        (
            "Backbone / head",
            (
                f"{getattr(encoder, 'backbone_name', type(encoder).__name__)} / "
                f"{(config.get('task') or {}).get('architecture', 'N/A')} (stage {stage})"
            ),
        ),
        ("Kiến trúc", architecture),
        ("Phương pháp", f"{CAM_METHODS[cam_method]}. {CAM_MEANING}"),
        ("Target layer", _describe_target_layer(encoder, hook_module, config, channels)),
        ("Pooling (đo từ gradient)", pooling_text),
        (
            "Mode",
            (
                "model.eval() + torch.enable_grad(); một backward từ logit, gradient được xóa sau "
                "đó; không optimizer step, không đổi weights"
            ),
        ),
        ("Input model", shape(input_shape)),
        ("CT hiển thị", f"{shape(display_shape)} ({canvas_source})"),
        ("Feature map trước nội suy", f"{channels} × {shape(grid_shape)}; mỗi ô = {cell_text}"),
        *display_rows(
            vmax,
            hu_known,
            ("bilinear trong mặt phẳng từng feature slice lên CT tham chiếu; mỗi ô sâu nhất "
             "của CT-FM nhận thông tin từ gần như cả patch 24×128×128 nên vị trí thật "
             "còn thô hơn lưới ô" if cached else
             "trilinear (align_corners=False) lên lưới CT"),
        ),
        ("Hướng / spacing hiển thị", f"{orientation_text}; spacing (S, P, L) = {spacing_text}"),
        ("Slice gốc", mapping_text),
        ("Tổng đóng góp bậc 1 (trước ReLU)", contribution_text(raw_cam)),
        ("CAM trong padding canvas (QC)", padding_text),
        ("CAM ngoài cơ thể (QC)", outside_text),
        ("Phân bố CAM dương theo ô z", profile_text),
        ("Kiểm tra xác suất", probability_check),
        ("Chọn ca", selection_text),
    ]


def _preview_indices(
    dataset: Any,
    maximum_patients: int,
    selection: Mapping[str, Any] | None,
    threshold: float | None,
    reference: Mapping[tuple[str, str], float] | None,
    target: str,
) -> tuple[list[int], str]:
    """Which validation studies to explain, and a one-line description of the rule.

    ``selection = {"correct": 3, "wrong": 3}`` picks the most confident correct cases
    (alternating TP / TN) and the most confident errors (alternating FP / FN), judged with
    evaluate's validation threshold and probabilities. Without them (training-time preview)
    it falls back to the first ``maximum_patients`` studies.
    """
    default = list(range(min(int(maximum_patients), len(dataset))))
    options = dict(selection or {})
    correct_n, wrong_n = int(options.get("correct", 0)), int(options.get("wrong", 0))
    base = _base_dataset(dataset)
    if not (correct_n or wrong_n) or threshold is None or not reference or base is None:
        return default, "các study đầu của validation theo thứ tự manifest; không chọn theo đúng/sai"
    pools: dict[str, list[tuple[float, int]]] = {"TP": [], "TN": [], "FP": [], "FN": []}
    for index, row in enumerate(base.rows):
        probability = reference.get((str(row.get("patient_id")), str(row.get("study_id"))))
        try:
            truth = float(row.get(target))
        except (TypeError, ValueError):
            continue
        if probability is None or not math.isfinite(truth) or not math.isfinite(float(probability)):
            continue
        outcome = _outcome(int(truth), int(float(probability) >= float(threshold)))
        pools[outcome].append((abs(float(probability) - float(threshold)), index))
    for pool in pools.values():
        pool.sort(reverse=True)

    def take(first: str, second: str, count: int) -> list[int]:
        chosen: list[int] = []
        queues = [list(pools[first]), list(pools[second])]
        turn = 0
        while len(chosen) < count and (queues[0] or queues[1]):
            queue = queues[turn % 2] if queues[turn % 2] else queues[(turn + 1) % 2]
            chosen.append(queue.pop(0)[1])
            turn += 1
        return chosen

    indices = take("TP", "TN", correct_n) + take("FP", "FN", wrong_n)
    counts = ", ".join(f"{name}={len(pool)}" for name, pool in pools.items())
    text = (
        f"{correct_n} ca đúng (xen kẽ TP/TN) + {wrong_n} ca sai (xen kẽ FP/FN) tự tin nhất theo |p − threshold| "
        f"trên validation (threshold của evaluate); số ca có sẵn: {counts}"
    )
    return indices, text


def _write_cam_nifti(
    stem: Path, ct: Any, cam: Any, metadata: Mapping[str, Any] | None, hu_known: bool
) -> list[str]:
    """``<stem>_ct.nii.gz`` + ``<stem>_gradcam.nii.gz`` on the model's input grid.

    Both carry the cache sidecar's ``output_affine`` (world coordinates of the input array),
    so they overlay in ITK-SNAP / 3D Slicer, and also on the original CT when the affine is
    known. The CAM is the positive map scaled to [0, 1] per study; CT is int16 HU when the
    cache recorded its HU window, float otherwise. Without a sidecar the affine is identity.
    """
    import nibabel as nib
    import numpy as np

    affine = np.eye(4)
    try:
        candidate = np.asarray((metadata or {}).get("output_affine"), dtype=float)
        if candidate.shape == (4, 4) and np.isfinite(candidate).all():
            affine = candidate
    except (TypeError, ValueError):
        pass
    ct_array = np.asarray(ct)
    ct_array = np.round(ct_array).astype(np.int16) if hu_known else ct_array.astype(np.float32)
    cam_array = np.asarray(cam, dtype=np.float32)
    peak = float(cam_array.max()) if cam_array.size else 0.0
    cam_array = cam_array / peak if peak > 0 else cam_array
    names = [f"{stem.name}_ct.nii.gz", f"{stem.name}_gradcam.nii.gz"]
    nib.save(nib.Nifti1Image(ct_array, affine), str(stem.parent / names[0]))
    nib.save(nib.Nifti1Image(cam_array, affine), str(stem.parent / names[1]))
    return names


def _write_attention_plot(destination: Path, metadata: Mapping[str, Any], title: str) -> Path | None:
    """Bar chart of the slice-MIL attention over z for one study (2D / 2.5D models)."""
    attention = metadata.get("slice_attention")
    indices = metadata.get("slice_indices")
    if attention is None or not indices:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = attention[0].float().cpu().numpy() if isinstance(attention, Tensor) else list(attention[0])
    top = sorted(range(len(weights)), key=lambda k: -float(weights[k]))[:5]
    figure, axis = plt.subplots(figsize=(10, 3.2), dpi=120)
    colors = ["#eb6834" if k in top else "#2a78d6" for k in range(len(weights))]
    axis.bar([int(value) for value in indices], weights, width=max(1.0, 0.8 * (indices[1] - indices[0]) if len(indices) > 1 else 1.0), color=colors)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_xlabel("chỉ số lát z trên volume input (RAS, dưới → trên)")
    axis.set_ylabel("trọng số MIL")
    axis.set_title(title + " · top-5: " + ", ".join(f"z={int(indices[k])}" for k in top), fontsize=9)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


def write_backbone_previews(
    model: nn.Module,
    dataset: Any,
    config: Mapping[str, Any],
    device: torch.device,
    destination: Path,
    *,
    maximum_patients: int = 5,
    threshold: float | None = None,
    threshold_rule: str | None = None,
    reference_probabilities: Mapping[tuple[str, str], float] | None = None,
    checkpoint: Path | str | None = None,
    checkpoint_sha256: str | None = None,
    cam_method: str = "hirescam",
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Grad-CAM preview of validation studies: which input regions drive the target.

    ``selection={"correct": 3, "wrong": 3}`` (``preview`` in the run config) explains the most
    confident correct and wrong validation cases instead of the first studies; slice-MIL
    models also get ``<case>_mil_attention.png`` (MIL weight per slice).

    For each study (the first ``maximum_patients`` of ``dataset``, not chosen by outcome)
    this writes ``NN_<patient>_<study>[_TP|TN|FP|FN].html``, an offline viewer with every
    input z level as CT reference | CT + Grad-CAM, and a ``.png`` summary with the 8 highest-CAM levels
    (see ``source/imaging/cam_preview.py``).

    The CAM explains the target logit of the final output: one forward pass in ``eval()``
    with gradients enabled, ``backward()`` from that logit, and activation/gradient of the
    deepest 5-D backbone feature map (the input of the trainable head for cached CT-FM
    features). Weights, preprocessing and prediction logic are untouched; the gradients are
    cleared afterwards and the training/eval mode is restored.

    ``threshold`` is the validation-selected cutoff of evaluate (never re-chosen here);
    without it (training phase) the predicted class is N/A. ``reference_probabilities``
    maps (patient_id, study_id) to evaluate's probability; the header shows it and the
    preview forward is checked against it. Validation-only, qualitative output.
    """
    import numpy as np

    from source.imaging.cam_preview import (
        MONTAGE_SLICES,
        cam_status,
        display_orientation,
        original_slice_mapping,
        outside_body_fraction,
        padding_fraction,
        rank_slices,
        slice_scores,
        top_slices,
        write_cam_preview,
    )
    from source.imaging.grid import read_sidecar

    if cam_method not in CAM_METHODS:
        raise ValueError(f"unknown CAM method {cam_method!r}; choose one of {sorted(CAM_METHODS)}")
    dataset = _grid_view(dataset)
    destination.mkdir(parents=True, exist_ok=True)
    # Only this contract's files belong in preview/; remove images and notes of earlier ones.
    for pattern in ("*.png", "*.html", "*.nii.gz"):
        for stale in destination.glob(pattern):
            stale.unlink()
    for name in LEGACY_PREVIEW_FILES:
        (destination / name).unlink(missing_ok=True)
    stage = str((config.get("experiment") or {}).get("stage") or "")
    if stage == "ablation":
        stage = str((config.get("task") or {}).get("base_stage") or stage)
    if stage not in {"diagnosis", "prognosis"}:
        return {"status": "skipped", "reason": f"stage_{stage or 'unknown'}", "patients": 0, "errors": []}
    target = str((config.get("task") or {}).get("primary_target") or "pe_present")
    label_columns = list((config.get("data") or {}).get("label_columns") or ())
    experiment_id = str((config.get("experiment") or {}).get("id") or "N/A")
    underlying = _unwrap(model)
    encoder = getattr(underlying, "image_encoder", None)
    if encoder is None:
        return {"status": "skipped", "reason": "no_image_backbone", "patients": 0, "errors": []}
    base_dataset = _base_dataset(dataset)
    dataset_transform = getattr(base_dataset, "transform", None)

    written = 0
    errors: list[str] = []
    outcomes: list[str] = []
    statuses: list[str] = []
    files: list[str] = []
    deltas: list[float] = []
    probabilities: list[float] = []
    hook_state: dict[str, Tensor] = {}

    def _feature_map_from_output(output: Any) -> Tensor | None:
        """Extract a spatial feature map from wrapper/backbone outputs."""
        feature_map = getattr(output, "feature_map", None)
        if feature_map is None and isinstance(output, Mapping):
            feature_map = output.get("feature_map")
        if isinstance(feature_map, Tensor) and feature_map.ndim == 5:
            return feature_map
        if isinstance(output, Tensor) and output.ndim == 5:
            return output
        if isinstance(output, (list, tuple)):
            for candidate in reversed(output):
                nested = _feature_map_from_output(candidate)
                if nested is not None:
                    return nested
        return None

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        feature_map = _feature_map_from_output(output)
        if feature_map is None:
            raise RuntimeError("backbone output did not expose a 5D feature_map")
        hook_state["feature_map"] = feature_map
        if isinstance(output, Mapping) and isinstance(output.get("metadata"), Mapping):
            hook_state["metadata"] = output["metadata"]  # type: ignore[assignment]
        if isinstance(feature_map, Tensor) and feature_map.requires_grad:
            feature_map.retain_grad()

    # BaseImageEncoder.forward() returns only global_embedding. Hook the actual
    # third-party backbone when the project wrapper exposes it, so the deepest feature map
    # (the Grad-CAM target layer) is available.
    hook_module = getattr(encoder, "model", None)
    if not isinstance(hook_module, nn.Module):
        hook_module = encoder
    target_for_selection = str((config.get("task") or {}).get("primary_target") or "pe_present")
    indices, selection_text = _preview_indices(
        dataset, maximum_patients, selection or (config.get("preview") or None),
        threshold, reference_probabilities, target_for_selection,
    )
    handle = hook_module.register_forward_hook(capture)
    was_training = underlying.training
    try:
        underlying.eval()
        for index in indices:
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
                # A leaf that requires grad keeps the graph alive even when every backbone
                # parameter is frozen; it is detached from the dataset tensor.
                model_input = batch["volume"].detach().float().requires_grad_(True)
                batch["volume"] = model_input
                underlying.zero_grad(set_to_none=True)
                hook_state.clear()
                # The unwrapped module: only rank 0 writes previews, and a DDP forward/backward
                # would start collectives the other ranks never join.
                with torch.enable_grad():
                    output = (
                        underlying(model_input, batch["masks"]) if stage == "diagnosis" else underlying(batch)
                    )
                    logit = _primary_logit(output, stage, target)
                    logit.backward()
                preview_probability = float(torch.sigmoid(logit.detach()).float().cpu())
                feature_map = hook_state.get("feature_map")
                if feature_map is None:
                    raise RuntimeError("backbone feature map was unavailable")
                gradient = feature_map.grad
                raw = None if gradient is None else cam_from_target_layer(feature_map, gradient, cam_method)
                raw_np = None if raw is None else raw.numpy()
                status, vmax, status_reason = cam_status(raw_np)
                grid_shape = tuple(int(value) for value in feature_map.shape[-3:])
                channels = int(feature_map.shape[1])
                warnings: list[str] = []
                notes: list[str] = []
                pooling = _pooling_from_gradient(gradient)

                image_path = _dataset_image_path(dataset, index)
                sidecar = read_sidecar(image_path) if image_path is not None else None
                if int(model_input.shape[1]) == 1:
                    display = model_input[0, 0].detach().float().cpu().numpy()
                    canvas_source = "tensor input của model"
                    metadata = sidecar
                    recorded = [int(value) for value in (sidecar or {}).get("output_shape") or ()][-3:]
                    if metadata and (recorded != list(display.shape) or dataset_transform is not None):
                        warnings.append(
                            "Dataset transform đã đổi tensor so với cache: hướng, spacing, HU và "
                            "slice gốc không xác định từ sidecar (N/A)."
                        )
                        metadata = None
                else:
                    # Cached CT-FM features are not an image; rebuild the CT canvas they were
                    # extracted from so the maps can be drawn on the anatomy.
                    display, rebuilt = _display_ct_for_features(image_path)
                    canvas_source = (
                        "canvas CT dựng lại từ NIfTI gốc theo spec của sidecar cache (đúng canvas "
                        "CT-FM nhận khi build cache)"
                    )
                    metadata = sidecar
                    mismatched = _metadata_mismatches(rebuilt, sidecar or {})
                    if mismatched:
                        warnings.append(
                            "Canvas dựng lại khác sidecar cache ở " + ", ".join(mismatched)
                            + ": CT có thể không khớp feature đã cache."
                        )
                display = np.asarray(display, dtype=np.float32)
                hu_known = False
                ct = display
                hu_range = (metadata or {}).get("hu_range")
                if metadata and str(metadata.get("normalization")) == "minmax" and hu_range:
                    low, high = (float(value) for value in hu_range)
                    ct = display * (high - low) + low
                    hu_known = True

                orientation, spacing = _volume_geometry(metadata)
                labels_text, orientation_text = display_orientation(
                    orientation, (metadata or {}).get("output_affine")
                )
                if labels_text is None:
                    orientation = None
                    orientation_text = f"N/A — {orientation_text}; hiển thị theo trục mảng 0, không xoay"
                size = tuple(int(value) for value in display.shape[-3:])
                arrays = [ct]
                nifti_payload = None
                if status == "ok":
                    positive = torch.from_numpy(np.clip(raw_np, 0.0, None)).float()
                    slice_mil = isinstance(hook_state.get("metadata"), Mapping) and (
                        hook_state["metadata"].get("slice_attention") is not None
                    )
                    if slice_mil and positive.shape[-1] < size[2]:
                        # Slice-MIL maps are independent 2-D slices stacked along z: upsample
                        # in-plane only, then hold each slice's CAM over its own z block instead
                        # of interpolating between slices the model never related.
                        in_plane = F.interpolate(
                            positive[None, None], size=(size[0], size[1], int(positive.shape[-1])),
                            mode="trilinear", align_corners=False,
                        )
                        cam_up = F.interpolate(in_plane, size=size, mode="nearest")[0, 0].numpy()
                    else:
                        cam_up = F.interpolate(
                            positive[None, None], size=size, mode="trilinear", align_corners=False
                        )[0, 0].numpy()
                    arrays += [cam_up, positive.numpy()]
                    # CT + CAM on the model's input grid (before display reorientation).
                    nifti_payload = (ct, cam_up, metadata, hu_known)
                    padding = padding_fraction(cam_up, metadata)
                    if padding is not None and padding > 0:
                        notes.append(
                            f"{100 * padding:.1f}% khối CAM dương nằm trong vùng padding của canvas "
                            "(không có dữ liệu quét)."
                            + (" Head pool đều trên toàn lưới feature nên các ô đó có đóng góp thật."
                               if pooling["uniform"] else "")
                        )
                else:
                    padding = None
                # Slice-MIL models: track where each input z lands in the displayed stack so
                # the montage can show the slices with the highest MIL attention.
                mil_metadata = hook_state.get("metadata")
                track_z = (
                    isinstance(mil_metadata, Mapping)
                    and mil_metadata.get("slice_attention") is not None
                    and int(model_input.shape[1]) == 1
                )
                if track_z:
                    arrays.append(np.broadcast_to(
                        np.arange(size[2], dtype=np.float32)[None, None, :], size
                    ).copy())
                oriented, display_spacing = _to_spl(arrays, orientation, spacing)
                display_z = None
                if track_z:
                    zmap = oriented.pop()
                    if np.allclose(zmap.std(axis=(1, 2)), 0.0):   # input z is the display axis
                        display_z = zmap.mean(axis=(1, 2))
                ct_display = oriented[0]
                cam_up_display = oriented[1] if status == "ok" else None
                cam_grid_display = oriented[2] if status == "ok" else None
                full_depth = int(ct_display.shape[0])
                full_mapping = (
                    original_slice_mapping(metadata, full_depth) if orientation is not None
                    else {"available": False, "reason": "hướng ảnh không xác định", "entries": []}
                )
                slice_labels = None
                if int(model_input.shape[1]) > 1:
                    # The diagnosis head receives 10 cached feature slices in this run,
                    # not 120 CT canvas slices. Show exactly one CT reference at each
                    # feature-cell centre, without implying voxel-level localization.
                    feature_depth = int(grid_shape[0])
                    if orientation not in (None, "SPL"):
                        raise RuntimeError("cached feature grid must be SPL to pair with the CT canvas")
                    if feature_depth < 1 or any(a % b for a, b in zip(
                        ct_display.shape, (feature_depth, grid_shape[1], grid_shape[2])
                    )):
                        raise RuntimeError("CT canvas does not align by integer cells to the cached feature grid")
                    centres = [min(full_depth - 1, round((z + .5) * full_depth / feature_depth - .5))
                               for z in range(feature_depth)]
                    ct_display = ct_display[centres]
                    if cam_grid_display is not None:
                        cam_up_display = F.interpolate(
                            torch.from_numpy(cam_grid_display.copy()).float().unsqueeze(1),
                            size=ct_display.shape[1:], mode="bilinear", align_corners=False,
                        )[:, 0].numpy()
                    if full_mapping.get("available"):
                        entries = full_mapping["entries"]
                        mapping = {**full_mapping, "entries": [
                            {**entries[canvas_z], "z": z} for z, canvas_z in enumerate(centres)
                        ]}
                    else:
                        mapping = full_mapping
                    slice_labels = [
                        {"input": f"feature slice {z + 1}/{feature_depth} (z={z}); CT canvas centre z={canvas_z}",
                         "slice": f"feature {z + 1}/{feature_depth}"}
                        for z, canvas_z in enumerate(centres)
                    ]
                    if display_spacing is not None:
                        display_spacing = (
                            display_spacing[0] * full_depth / feature_depth,
                            display_spacing[1], display_spacing[2],
                        )
                    canvas_source += (
                        f"; viewer lấy {feature_depth} lát CT tham chiếu tại tâm ô z của "
                        f"canvas {full_depth} lát; model chỉ nhận feature, không nhận trực tiếp CT này"
                    )
                    size = tuple(int(value) for value in ct_display.shape)
                else:
                    mapping = full_mapping
                scores = ranks = None
                top: list[int] = []
                outside = None
                if status == "ok":
                    scores = slice_scores(cam_up_display)
                    ranks, _ = rank_slices(scores)
                    top = top_slices(scores, MONTAGE_SLICES)
                    if display_z is not None:
                        weights = mil_metadata["slice_attention"][0].float().cpu().numpy()
                        chosen = [int(mil_metadata["slice_indices"][k]) for k in np.argsort(-weights)[:MONTAGE_SLICES]]
                        attention_top = list(dict.fromkeys(int(np.argmin(np.abs(display_z - z))) for z in chosen))
                        if attention_top:
                            top = attention_top
                            notes.append(
                                f"Montage slice-MIL: {len(top)} lát có trọng số attention MIL cao nhất "
                                f"(z input = {', '.join(str(z) for z in chosen)}), không chọn theo điểm CAM; "
                                "xem thêm _mil_attention.png."
                            )
                    if hu_known:
                        outside = outside_body_fraction(ct_display, cam_up_display)

                truth: int | None = None
                labels = item.get("labels")
                if isinstance(labels, Tensor) and target in label_columns:
                    value = float(labels.reshape(-1)[label_columns.index(target)])
                    truth = int(value) if np.isfinite(value) else None
                reference = (reference_probabilities or {}).get((patient, study))
                probability = float(reference) if reference is not None else preview_probability
                delta = abs(preview_probability - float(reference)) if reference is not None else None
                if delta is not None:
                    deltas.append(delta)
                    if delta > PROBABILITY_TOLERANCE:
                        warnings.append(
                            f"Xác suất forward của preview ({preview_probability:.6f}) lệch evaluate "
                            f"({float(reference):.6f}) {delta:.2e} > {PROBABILITY_TOLERANCE:g}: CAM có thể "
                            "không giải thích đúng dự đoán đã báo cáo."
                        )
                predicted = None
                if threshold is not None and math.isfinite(probability):
                    predicted = int(probability >= float(threshold))
                elif not math.isfinite(probability):
                    warnings.append("Xác suất không hữu hạn (NaN/Inf): không gán nhãn dự đoán.")
                outcome = _outcome(truth, predicted)
                if status != "ok":
                    warnings.append(f"CAM {status}: {status_reason}.")

                def label_text(value: int | None) -> str:
                    return "N/A" if value is None else ("positive (1)" if value else "negative (0)")

                probability_name = "Xác suất PE" if target == "pe_present" else f"Xác suất {target}"
                header = [
                    ("patient_id", patient),
                    ("study_id", study),
                    ("Target giải thích", f"{target} — logit đầu ra cuối cùng"),
                    ("Ground truth", label_text(truth)),
                    ("Predicted", label_text(predicted) if threshold is not None
                     else "N/A (chưa có threshold; evaluate chọn trên validation)"),
                    ("TP/TN/FP/FN", outcome or "N/A"),
                    (probability_name, f"{probability:.6f}"),
                    ("Threshold", "N/A" if threshold is None else f"{float(threshold):.6f}"),
                    ("Cách chọn threshold", _threshold_rule_text(threshold_rule) if threshold is not None else "N/A"),
                    ("p − threshold", "N/A" if threshold is None else f"{probability - float(threshold):+.6f}"),
                ]
                technical = _technical_rows(
                    experiment_id=experiment_id,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    config=config,
                    stage=stage,
                    encoder=encoder,
                    hook_module=hook_module,
                    cam_method=cam_method,
                    input_shape=tuple(int(value) for value in model_input.shape),
                    display_shape=size,
                    canvas_source=canvas_source,
                    channels=channels,
                    grid_shape=grid_shape,
                    spacing=spacing,
                    vmax=vmax,
                    hu_known=hu_known,
                    orientation_text=orientation_text,
                    display_spacing=display_spacing,
                    mapping=mapping,
                    raw_cam=None if status == "invalid" else raw_np,
                    outside=outside,
                    padding=padding,
                    pooling_text=pooling["text"],
                    z_profile=(
                        None if cam_grid_display is None
                        else cam_grid_display.sum(axis=(1, 2)) / max(float(cam_grid_display.sum()), 1e-30)
                    ),
                    oriented=orientation is not None,
                    reference=reference,
                    preview_probability=preview_probability,
                    selection_text=selection_text,
                )
                prefix = f"{written + 1:02d}_{_safe_name(patient)}_{_safe_name(study)}"
                if outcome:
                    prefix += f"_{outcome}"
                metadata_hook = hook_state.get("metadata")
                if isinstance(metadata_hook, Mapping):
                    attention_file = _write_attention_plot(
                        destination / f"{prefix}_mil_attention.png", metadata_hook,
                        f"MIL attention · {patient} / {study} · {outcome or 'N/A'} · p={probability:.3f}",
                    )
                    if attention_file is not None:
                        files.append(attention_file.name)
                if nifti_payload is not None and bool((config.get("preview") or {}).get("nifti", False)):
                    files += _write_cam_nifti(destination / prefix, *nifti_payload)
                write_cam_preview(
                    destination / prefix,
                    {
                        "title": f"Grad-CAM {target} · {patient} / {study}",
                        "ct": ct_display,
                        "hu_known": hu_known,
                        "cam_up": cam_up_display,
                        "cam_grid": cam_grid_display,
                        "status": status,
                        "status_reason": status_reason,
                        "vmax": vmax,
                        "scores": scores,
                        "ranks": ranks,
                        "top": top,
                        "mapping": mapping,
                        "orientation": labels_text,
                        "orientation_text": orientation_text,
                        "spacing": display_spacing,
                        "header": header,
                        "technical": technical,
                        "warnings": warnings,
                        "notes": notes,
                        "slice_labels": slice_labels,
                        "ct_caption": ("CT tham chiếu tại tâm feature cell (model nhận feature, không nhận CT)"
                                       if int(model_input.shape[1]) > 1 else "CT trên lưới input model"),
                    },
                )
                files += [f"{prefix}.html", f"{prefix}.png"]
                probabilities.append(preview_probability)
                outcomes.append(outcome or "pending")
                statuses.append(status)
                written += 1
            except Exception as exc:  # preview is diagnostic; one bad case must not erase metrics
                errors.append(f"{patient}/{study}: {type(exc).__name__}: {exc}")
    finally:
        handle.remove()
        underlying.zero_grad(set_to_none=True)
        underlying.train(was_training)

    return {
        "status": "completed" if not errors else "completed_with_errors",
        "patients": written,
        "errors": errors,
        "outcomes": outcomes,
        "cam_status": statuses,
        "files": files,
        "method": cam_method,
        "max_abs_probability_delta": max(deltas) if deltas else None,
        "preview_probabilities": probabilities,
    }


def preview_log_lines(report: Mapping[str, Any], destination: Path) -> list[str]:
    """Run-log lines for a preview report (the preview folder itself holds only the viewers)."""
    outcomes = report.get("outcomes") or []
    statuses = report.get("cam_status") or []
    delta = report.get("max_abs_probability_delta")
    lines = [
        f"preview status={report.get('status')} patients={report.get('patients')} "
        f"dir={destination}"
        + (f" method={report['method']}" if report.get("method") else "")
        + (f" outcomes={','.join(outcomes)}" if outcomes else "")
        + (f" cam={','.join(statuses)}" if statuses else "")
        + (f" max_abs_dp_vs_evaluate={delta:.2e}" if delta is not None else "")
        + (f" reason={report['reason']}" if report.get("reason") else "")
    ]
    lines.extend(f"preview error {error}" for error in report.get("errors") or ())
    return lines


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
    "best_epoch",
    "cleanup_task_run",
    "latest_epoch_directory",
    "preview_log_lines",
    "refresh_epoch_log",
    "write_backbone_previews",
    "write_training_artifacts",
]
