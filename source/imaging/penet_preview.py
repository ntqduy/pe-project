"""Grad-CAM previews for the released PENet zero-shot baseline.

PENet scores a series as non-overlapping 32-slice windows (1 x 32 x 208 x 208 after its own
resize, crop and HU window) and aggregates the window probabilities, max by default. The
preview re-runs the loaded model on the first validation studies and explains
y = logit(series probability):

* per window, Grad-CAM at ``encoders[-1]`` (2048 x 2 x 7 x 7), the last feature map before
  ``GAPLinear``; with global average pooling the element-wise form equals classic Grad-CAM;
* through the aggregation: ``∂y/∂logit_w`` is 1 for the deciding window and 0 for every other
  window under max, and ``p_w (1 - p_w) / (n p̄ (1 - p̄))`` under mean, so a window's map is
  scaled by how much it moves the series output.

The viewer is the same as for trained models (``source/imaging/cam_preview.py``): every
slice PENet received, CT | CT + CAM, one normalisation per series, 8-slice montage.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from source.components.encoders.image.penet_zeroshot import (
    AIR_HU_VAL,
    CROP_SHAPE,
    NUM_SLICES,
    PENET_WEIGHT_SHA256,
    RESIZE_SHAPE,
    _normalize,
    _resize_and_crop,
    read_series_hu,
)
from source.imaging.cam_preview import (
    CAM_MEANING,
    CAM_METHODS,
    MONTAGE_SLICES,
    cam_from_target_layer,
    cam_status,
    contribution_text,
    display_rows,
    format_number,
    outside_body_fraction,
    rank_slices,
    slice_scores,
    top_slices,
    write_cam_preview,
)

# Preview forward vs the stored zero-shot probability; larger means another computation.
PROBABILITY_TOLERANCE = 1e-4


def series_inputs(volume: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """(display HU [W*32, 208, 208], model inputs [(1, 32, 208, 208)] * W) of one series.

    The same padding, resize and crop as ``penet_zeroshot.series_windows``; the display keeps
    HU before PENet's clip/normalisation so the CT can be windowed for reading.
    """
    total = volume.shape[0]
    count = total // NUM_SLICES + (1 if total % NUM_SLICES else 0)
    displays, inputs = [], []
    for index in range(count):
        window = volume[index * NUM_SLICES : (index + 1) * NUM_SLICES]
        missing = NUM_SLICES - window.shape[0]
        if missing > 0:
            window = np.pad(window, ((0, missing), (0, 0), (0, 0)), constant_values=AIR_HU_VAL)
        hounsfield = _resize_and_crop(window)
        displays.append(np.asarray(hounsfield, dtype=np.float32))
        inputs.append(_normalize(hounsfield)[None].astype(np.float32))
    return np.concatenate(displays, axis=0), inputs


def aggregation_weights(logits: Sequence[float], aggregate: str) -> tuple[float, np.ndarray, int]:
    """(series probability, ∂ logit(p_series) / ∂ logit_w, deciding window) for max/mean."""
    values = np.asarray(logits, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-values))
    deciding = int(np.argmax(probabilities))
    if aggregate == "max":
        weights = np.zeros_like(values)
        weights[deciding] = 1.0
        return float(probabilities[deciding]), weights, deciding
    if aggregate == "mean":
        mean = float(probabilities.mean())
        denominator = max(len(values) * mean * (1.0 - mean), 1e-12)
        return mean, probabilities * (1.0 - probabilities) / denominator, deciding
    raise ValueError(f"unknown aggregate {aggregate!r}")


def explain_series(
    model: Any,
    inputs: Sequence[np.ndarray],
    device: Any,
    *,
    aggregate: str,
    method: str,
) -> dict[str, Any]:
    """Per-window logits and Grad-CAM at ``model.encoders[-1]``, combined for the series.

    ``torch.autograd.grad`` returns ∂logit/∂A only; no parameter ``.grad`` is written.
    """
    import torch

    layer = model.encoders[-1]
    state: dict[str, Any] = {}
    handle = layer.register_forward_hook(lambda _module, _inputs, output: state.__setitem__("A", output))
    logits: list[float] = []
    maps: list[Any] = []
    channels = -1
    was_training = model.training
    try:
        model.eval()
        for window in inputs:
            tensor = torch.from_numpy(window).unsqueeze(0).to(device).requires_grad_(True)
            state.clear()
            with torch.enable_grad():
                logit = model(tensor).flatten()[0]
                (gradient,) = torch.autograd.grad(logit, state["A"])
            maps.append(cam_from_target_layer(state["A"], gradient, method).numpy())
            logits.append(float(logit.detach().float().cpu()))
            channels = int(state["A"].shape[1])
    finally:
        handle.remove()
        model.train(was_training)
    probability, weights, deciding = aggregation_weights(logits, aggregate)
    raw = np.concatenate([weight * cam for weight, cam in zip(weights, maps)], axis=0)
    return {
        "logits": logits,
        "probability": probability,
        "weights": weights,
        "deciding": deciding,
        "raw": raw,
        "window_grid": tuple(int(value) for value in maps[0].shape),
        "channels": channels,
    }


def original_slice_mapping(path: Path, total: int, depth: int, slice_order: str) -> dict[str, Any]:
    """Model input slice s -> slice of the original NIfTI array (index exact: PENet does not
    resample along z). Slices past ``total`` are PENet's air padding."""
    import nibabel as nib

    image = nib.load(str(path))
    affine = np.asarray(image.affine, dtype=float)
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(affine), nib.orientations.axcodes2ornt(("R", "A", "S"))
    )
    source_axis = next(index for index, (target, _) in enumerate(transform) if int(target) == 2)
    flipped = float(transform[source_axis][1]) < 0
    length = int(image.shape[source_axis])
    entries: list[dict[str, Any]] = []
    for s in range(depth):
        if s >= total:
            entries.append({"z": s, "status": "padding"})
            continue
        canonical = total - 1 - s if slice_order == "superior_to_inferior" else s
        original = length - 1 - canonical if flipped else canonical
        entries.append(
            {"z": s, "status": "mapped", "position": float(original), "lower": original, "upper": original}
        )
    return {
        "available": True,
        "reason": "",
        "axis": int(source_axis),
        "axis_name": "ijk"[source_axis],
        "length": length,
        "original_spacing_mm": float(np.linalg.norm(affine[:3, source_axis])),
        "entries": entries,
    }


def _geometry(path: Path, volume_shape: Sequence[int]) -> tuple[tuple[float, float, float], float]:
    """(display spacing slice, row, column in mm, affine obliquity in degrees)."""
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(str(path)))
    sx, sy, sz = (float(value) for value in image.header.get_zooms()[:3])
    _, rows, columns = (int(value) for value in volume_shape)
    # cv2.resize to RESIZE_SHAPE, then a centre crop that keeps the pixel size.
    spacing = (sz, sy * rows / RESIZE_SHAPE[1], sx * columns / RESIZE_SHAPE[0])
    obliquity = float(np.degrees(np.max(nib.affines.obliquity(nib.load(str(path)).affine))))
    return spacing, obliquity


def _outcome(truth: int, predicted: int) -> str:
    return ("TP" if truth else "FP") if predicted else ("FN" if truth else "TN")


def _render(
    model: Any,
    device: Any,
    path: Path,
    destination_stem: Path,
    row: Mapping[str, Any],
    *,
    threshold: float,
    threshold_rule: str,
    slice_order: str,
    aggregate: str,
    method: str,
    checkpoint: str,
    target: str,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    volume = read_series_hu(path, slice_order)
    if not np.isfinite(volume).all():
        raise ValueError(f"CT contains non-finite voxels: {path}")
    total = int(volume.shape[0])
    display, inputs = series_inputs(volume)
    explained = explain_series(model, inputs, device, aggregate=aggregate, method=method)
    raw, windows = explained["raw"], len(inputs)
    status, vmax, status_reason = cam_status(raw)
    depth = int(display.shape[0])
    per_window = explained["window_grid"]
    cam_grid = cam_up = scores = ranks = None
    top: list[int] = []
    padding = outside = None
    if status == "ok":
        cam_grid = np.clip(raw, 0.0, None)
        # Windows run independently: upsample each one on its own, never across a boundary.
        cam_up = np.concatenate([
            F.interpolate(
                torch.from_numpy(cam_grid[index * per_window[0] : (index + 1) * per_window[0]].copy())
                .float()[None, None],
                size=(NUM_SLICES, *CROP_SHAPE), mode="trilinear", align_corners=False,
            )[0, 0].numpy()
            for index in range(windows)
        ], axis=0)
        scores = slice_scores(cam_up)
        ranks, _ = rank_slices(scores)
        top = top_slices(scores, MONTAGE_SLICES)
        total_mass = float(cam_up.sum())
        padding = float(cam_up[total:].sum() / total_mass) if total_mass > 0 else None
        outside = outside_body_fraction(display, cam_up)
    mapping = original_slice_mapping(path, total, depth, slice_order)
    spacing, obliquity = _geometry(path, volume.shape)

    truth = int(row["y_true"])
    reference = float(row["y_prob"])
    preview_probability = explained["probability"]
    delta = abs(preview_probability - reference)
    predicted = int(reference >= float(threshold))
    outcome = _outcome(truth, predicted)
    warnings: list[str] = []
    notes: list[str] = []
    if delta > PROBABILITY_TOLERANCE:
        warnings.append(
            f"Xác suất forward của preview ({preview_probability:.6f}) lệch kết quả zero-shot "
            f"({reference:.6f}) {delta:.2e}: CAM có thể không giải thích đúng dự đoán đã báo cáo."
        )
    if status != "ok":
        warnings.append(f"CAM {status}: {status_reason}.")
    deciding = explained["deciding"]
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(explained["logits"])))
    if aggregate == "max" and windows > 1:
        notes.append(
            f"Xác suất series = max qua {windows} cửa sổ: chỉ cửa sổ {deciding + 1} "
            f"(slice {deciding * NUM_SLICES + 1}–{(deciding + 1) * NUM_SLICES}) có gradient tới "
            "đầu ra, nên CAM của các cửa sổ khác bằng 0 theo định nghĩa, không phải vì model "
            "không thấy gì ở đó."
        )
    if padding:
        notes.append(f"{100 * padding:.1f}% khối CAM dương nằm trên slice không khí PENet đệm thêm.")
    orientation_text = (
        "PENet đọc NIfTI qua as_closest_canonical rồi xếp như DICOM: trước (A) ở trên, trái bệnh "
        "nhân (L) bên phải ảnh; slice đầu vào 1 = phía "
        + ("trên (superior)" if slice_order == "superior_to_inferior" else "dưới (inferior)")
    )
    if obliquity > 0.5:
        orientation_text += f"; affine xiên {obliquity:.1f}°: nhãn theo trục gần nhất"
    labels = {"top": "A", "bottom": "P", "left": "R", "right": "L"}
    end = "trên/superior" if slice_order == "superior_to_inferior" else "dưới/inferior"
    slice_labels = []
    for s in range(depth):
        window = s // NUM_SLICES
        decided = ", quyết định max" if aggregate == "max" and window == deciding else ""
        text = (
            f"{s + 1}/{depth} (cửa sổ {window + 1}/{windows}, p = {probabilities[window]:.3f}{decided}; "
            f"slice {s % NUM_SLICES + 1}/{NUM_SLICES}; 1 = phía {end})"
        )
        if s >= total:
            text += " — không khí PENet đệm thêm"
        slice_labels.append({"input": text, "slice": f"{s + 1}/{depth}"})

    header = [
        ("patient_id", str(row["patient_id"])),
        ("study_id", str(row["study_id"])),
        ("Target giải thích", f"{target} — logit của xác suất series (sau bước gộp {aggregate})"),
        ("Ground truth", "positive (1)" if truth else "negative (0)"),
        ("Predicted", "positive (1)" if predicted else "negative (0)"),
        ("TP/TN/FP/FN", outcome),
        ("Xác suất PE" if target == "pe_present" else f"Xác suất {target}", f"{reference:.6f}"),
        ("Threshold", f"{float(threshold):.6f}"),
        ("Cách chọn threshold", _threshold_text(threshold_rule)),
        ("p − threshold", f"{reference - float(threshold):+.6f}"),
    ]
    cells = (NUM_SLICES / per_window[0], CROP_SHAPE[0] / per_window[1], CROP_SHAPE[1] / per_window[2])
    cell_text = (
        " × ".join(format_number(value, 4) for value in cells) + " pixel = "
        + " × ".join(format_number(value * step, 4) for value, step in zip(cells, spacing))
        + " mm (slice × hàng × cột); 208/7 không chia hết nên ánh xạ ô → pixel theo hàng/cột chỉ gần đúng"
    )
    aggregation_text = (
        "max: ∂y/∂logit_w = 1 ở cửa sổ quyết định, 0 ở mọi cửa sổ khác"
        if aggregate == "max"
        else "mean: ∂y/∂logit_w = p_w(1 − p_w) / (n·p̄(1 − p̄))"
    )
    technical = [
        ("Model", "PENetClassifier, trọng số phát hành, không train lại (zero-shot)"),
        ("Checkpoint", checkpoint or "N/A"),
        ("Checkpoint SHA-256", PENET_WEIGHT_SHA256),
        (
            "Kiến trúc",
            (
                f"CNN 3D trên từng cửa sổ {NUM_SLICES} slice liên tiếp không chồng lấn (1×{NUM_SLICES}×"
                f"{CROP_SHAPE[0]}×{CROP_SHAPE[1]}, nhân lên 3 kênh bên trong); xác suất series = "
                f"{aggregate} của sigmoid(logit) qua {windows} cửa sổ"
            ),
        ),
        ("Gradient qua bước gộp", f"y = logit(p_series); {aggregation_text}"),
        (
            "Phương pháp",
            f"{CAM_METHODS[method]}. GAPLinear pool trung bình đều nên HiResCAM = Grad-CAM gốc. {CAM_MEANING}",
        ),
        (
            "Target layer",
            f"encoders[-1] (PENetEncoder cuối, {explained['channels']} kênh), tầng cuối trước GAPLinear",
        ),
        (
            "Mode",
            (
                "model.eval() + torch.enable_grad(); torch.autograd.grad chỉ lấy ∂logit/∂A, không ghi "
                ".grad của weights, không đổi weights"
            ),
        ),
        ("Input model", f"{windows} × 1 × {NUM_SLICES} × {CROP_SHAPE[0]} × {CROP_SHAPE[1]}"),
        (
            "CT hiển thị",
            (
                f"{depth} × {CROP_SHAPE[0]} × {CROP_SHAPE[1]}: đúng hình học PENet nhận (resize "
                f"{RESIZE_SHAPE[0]} INTER_AREA, crop giữa {CROP_SHAPE[0]}, đệm không khí đến bội số "
                f"{NUM_SLICES}); cường độ là HU trước khi PENet clip [-100, 900]"
            ),
        ),
        (
            "Feature map trước nội suy",
            (
                f"{explained['channels']} × {' × '.join(str(v) for v in per_window)} mỗi cửa sổ → lưới "
                f"series {raw.shape[0]} × {per_window[1]} × {per_window[2]}; mỗi ô = {cell_text}"
            ),
        ),
        *display_rows(
            vmax, True, "trilinear (align_corners=False) trong từng cửa sổ, không qua ranh giới cửa sổ"
        ),
        (
            "Hướng / spacing hiển thị",
            f"{orientation_text}; spacing (slice, hàng, cột) = "
            + " × ".join(format_number(value, 4) for value in spacing) + " mm",
        ),
        (
            "Slice gốc",
            (
                f"{total}/{depth} slice đầu vào có dữ liệu quét; PENet không resample theo z nên slice "
                f"đầu vào ↔ slice gốc là 1:1 (trục {mapping['axis_name']}, dày "
                f"{format_number(mapping['original_spacing_mm'], 4)} mm)"
            ),
        ),
        (
            "Xác suất từng cửa sổ",
            " · ".join(f"{index + 1}: {value:.3f}" for index, value in enumerate(probabilities)),
        ),
        ("Tổng đóng góp bậc 1 (trước ReLU)", contribution_text(None if status == "invalid" else raw)),
        ("CAM trên slice đệm (QC)", "N/A" if padding is None else f"{100 * padding:.1f}% khối CAM dương"),
        (
            "CAM ngoài cơ thể (QC)",
            "N/A" if outside is None else f"{100 * outside:.1f}% khối CAM dương nằm ngoài mask cơ thể thô "
            "(HU > −500, thành phần lớn nhất, lấp lỗ từng slice); mask không che heatmap",
        ),
        (
            "Kiểm tra xác suất",
            f"zero-shot {reference:.6f} vs forward preview {preview_probability:.6f}, |Δ| = {delta:.2e}",
        ),
        ("Chọn ca", "các study đầu của validation theo thứ tự kết quả; không chọn theo đúng/sai"),
    ]
    write_cam_preview(
        destination_stem,
        {
            "title": f"Grad-CAM {target} · PENet zero-shot · {row['patient_id']} / {row['study_id']}",
            "ct": display,
            "hu_known": True,
            "cam_up": cam_up,
            "cam_grid": cam_grid,
            "status": status,
            "status_reason": status_reason,
            "vmax": vmax,
            "scores": scores,
            "ranks": ranks,
            "top": top,
            "mapping": mapping,
            "orientation": labels,
            "orientation_text": orientation_text,
            "spacing": spacing,
            "header": header,
            "technical": technical,
            "warnings": warnings,
            "notes": notes,
            "slice_labels": slice_labels,
            "z_blocks": windows if status == "ok" else 1,
        },
    )
    return {"status": status, "outcome": outcome, "delta": delta, "probability": preview_probability}


def _threshold_text(rule: str) -> str:
    text = str(rule or "")
    if text.endswith("_on_validation"):
        method = text.removesuffix("_on_validation")
        detail = "Youden J = sensitivity + specificity − 1 cực đại" if method == "youden" else method
        return f"{text}: {detail} trên validation, chọn bởi zero-shot tool; preview không chọn lại"
    if text == "default_0.5_validation_one_class":
        return f"{text}: 0.5 mặc định vì validation chỉ có một lớp"
    return text or "N/A"


def write_penet_previews(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    raw_series_path: Callable[[str], Path],
    *,
    model: Any,
    device: Any,
    threshold: float,
    threshold_rule: str,
    maximum_patients: int = 5,
    slice_order: str = "superior_to_inferior",
    aggregate: str = "max",
    checkpoint: str | Path | None = None,
    target: str = "pe_present",
    cam_method: str = "hirescam",
) -> dict[str, Any]:
    """Grad-CAM viewer (.html) + summary (.png) for the first N validation studies.

    Uses the first validation rows, matching trained-model preview selection. ``rows`` carry
    the stored zero-shot ``y_prob``: it is what the header reports, and the preview's own
    forward is checked against it. The threshold is the one the zero-shot run chose.
    """
    if maximum_patients < 0:
        raise ValueError("maximum_patients must be nonnegative")
    if cam_method not in CAM_METHODS:
        raise ValueError(f"unknown CAM method {cam_method!r}")
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.html"):
        for stale in destination.glob(pattern):
            stale.unlink()
    selected = [row for row in rows if row.get("split") in (None, "validation")][:maximum_patients]
    written: list[str] = []
    errors: list[str] = []
    statuses: list[str] = []
    deltas: list[float] = []
    for index, row in enumerate(selected, 1):
        patient, study = str(row["patient_id"]), str(row["study_id"])
        truth = int(row["y_true"])
        predicted = int(float(row["y_prob"]) >= float(threshold))
        stem = f"{index:02d}_{patient}_{study}_{_outcome(truth, predicted)}"
        try:
            result = _render(
                model, device, raw_series_path(study), destination / stem, row,
                threshold=threshold, threshold_rule=threshold_rule, slice_order=slice_order,
                aggregate=aggregate, method=cam_method,
                checkpoint=str(checkpoint) if checkpoint else "", target=target,
            )
            written += [f"{stem}.html", f"{stem}.png"]
            statuses.append(result["status"])
            deltas.append(result["delta"])
        except Exception as exc:  # noqa: BLE001 - preview is diagnostic; never discard model results
            errors.append(f"{patient}/{study}: {type(exc).__name__}: {exc}")
    return {
        "status": "completed" if not errors else "completed_with_errors",
        "patients": len(statuses),
        "files": written,
        "errors": errors,
        "cam_status": statuses,
        "max_abs_probability_delta": max(deltas) if deltas else None,
        "method": cam_method,
        "selection": "first_validation_studies",
        "maximum_patients": maximum_patients,
        "content": "Grad-CAM at PENet encoders[-1], gradient through the window aggregation; "
        "html viewer of every input slice + png montage",
    }


def main() -> int:
    """Rebuild the preview of an existing PENet run (reruns the model, not the evaluation)."""
    import torch

    from source.components.encoders.image.penet_zeroshot import load_penet

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True, help="INSPECT CT/full/CTPA directory")
    parser.add_argument("--maximum-patients", type=int, default=5)
    parser.add_argument("--output", type=Path, help="default: <run-dir>/preview")
    parser.add_argument("--checkpoint", type=Path, help="default: the checkpoint recorded in result.json")
    parser.add_argument("--repo", type=Path, help="default: the repository recorded in result.json")
    parser.add_argument("--method", choices=sorted(CAM_METHODS), default="hirescam")
    parser.add_argument("--device", default=None, help="torch device (default: cuda when available)")
    args = parser.parse_args()
    result = json.loads((args.run_dir / "result.json").read_text(encoding="utf-8"))
    evaluation, zero_shot = result["evaluation"], result.get("zero_shot") or {}
    checkpoint = args.checkpoint or Path(str(zero_shot["checkpoint"]))
    repo = args.repo or Path(str(zero_shot["repo"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_penet(checkpoint, repo)
    model.to(device)
    with (args.run_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report = write_penet_previews(
        rows, args.output or args.run_dir / "preview", lambda study: args.raw_root / f"{study}.nii.gz",
        model=model,
        device=device,
        threshold=float(evaluation["threshold"]),
        threshold_rule=str(evaluation.get("threshold_rule") or ""),
        maximum_patients=args.maximum_patients,
        slice_order=str(zero_shot.get("slice_order") or "superior_to_inferior"),
        aggregate=str(zero_shot.get("aggregate") or "max"),
        checkpoint=checkpoint,
        target=str(evaluation.get("primary_target") or "pe_present"),
        cam_method=args.method,
    )
    if args.output is None:
        # result.json records the preview report of the run; keep it in step with preview/.
        result["preview"] = report
        temporary = args.run_dir / ".result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(args.run_dir / "result.json")
    print(json.dumps(report, indent=2))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
