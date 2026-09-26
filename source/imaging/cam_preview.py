"""Grad-CAM preview of one study: every input z level as CT reference | CT + CAM.

``write_cam_preview`` writes two files per study:

* ``<stem>.html``: self-contained viewer (no network). A slider walks through every axial
  input z level the model received, each shown with a geometrically aligned CT reference and CAM;
  opacity and display interpolation are adjustable; a montage shows the 8 slices with the
  highest CAM score; the header carries the prediction, the per-slice line the slice index,
  the original NIfTI slice, the CAM score/rank and the orientation.
* ``<stem>.png``: static summary (header + the same montage + colorbar) for a quick look.

Display rules, kept identical in both files:

* The CAM is normalised once per volume (positive CAM / its maximum on the feature grid) and
  drawn with one colormap and one colorbar; slices are never rescaled individually.
* Overlay alpha follows positive CAM magnitude, including air outside the body: no
  threshold or body/lung mask, so signal in implausible places stays visible.
* A CAM that is all zero or non-finite is reported as such and not drawn.
* Slice scores (mean positive CAM per slice) are computed on the raw CAM, before display
  normalisation, from the trilinear upsampling that is also the default display.

The geometry helpers only report what the preprocessing sidecar determines; anything else is
``N/A``.
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

MONTAGE_SLICES = 8
DEFAULT_OPACITY = 0.4
# CTPA vascular window (W/L 700/100 HU), the window every other CT preview of the project uses.
DEFAULT_WINDOW = (700.0, 100.0)
COLORMAP = "inferno"
# Positive CAM maxima at or below this are treated as "no positive contribution".
ZERO_TOLERANCE = 1e-12

DISCLAIMER = (
    "CAM thể hiện đóng góp tương đối theo phương pháp Grad-CAM cho logit của target được "
    "giải thích. Nó không phải xác suất PE của từng pixel/slice, không phải mask huyết khối "
    "và không phải bằng chứng model định vị đúng."
)
MONTAGE_NOTE = (
    "Montage được chọn theo điểm CAM (trung bình CAM dương của slice), không phải các slice "
    "đã được xác nhận có PE."
)


# ---------------------------------------------------------------------------------------
# CAM status, slice scores, ranks
# ---------------------------------------------------------------------------------------


# Grad-CAM variants over the target layer activation A (C x d x h x w) and y = target logit.
CAM_METHODS = {
    "hirescam": (
        "Grad-CAM dạng element-wise (HiResCAM): CAM = ReLU(Σ_k ∂y/∂A_k ⊙ A_k). Trùng Grad-CAM "
        "gốc khi head pooling trung bình đều; với pooling có trọng số (body coverage, ROI mask) "
        "nó chỉ gán đóng góp cho ô model thực sự đọc"
    ),
    "gradcam": (
        "Grad-CAM gốc (Selvaraju 2017): CAM = ReLU(Σ_k α_k A_k), α_k = trung bình không gian "
        "của ∂y/∂A_k"
    ),
}
# Appended to the method row: what the map is and is not.
CAM_MEANING = (
    "Đây là bản đồ đóng góp cho logit (có gradient của target), không phải feature activation "
    "(độ lớn activation, như nhau với mọi target); preview không vẽ activation"
)


def cam_from_target_layer(feature_map: Any, gradient: Any, method: str) -> Any:
    """Pre-ReLU CAM [d, h, w] of batch item 0 from target-layer activation and ∂y/∂A
    (torch tensors), in float64 on the CPU."""
    import torch

    activation = feature_map.detach()[0].double().cpu()
    grad = gradient.detach()[0].double().cpu()
    if method == "hirescam":
        return (grad * activation).sum(dim=0)
    if method == "gradcam":
        return torch.einsum("c,cdhw->dhw", grad.mean(dim=(1, 2, 3)), activation)
    raise ValueError(f"unknown CAM method {method!r}; choose one of {sorted(CAM_METHODS)}")


def cam_status(raw_cam: np.ndarray | None) -> tuple[str, float, str]:
    """(status, positive maximum, reason) of a pre-ReLU CAM on the feature grid.

    ``ok``: finite with a positive maximum; ``all_zero``: no positive value above
    ``ZERO_TOLERANCE``; ``invalid``: missing or containing NaN/Inf.
    """
    if raw_cam is None:
        return "invalid", float("nan"), "không có CAM (gradient không tới target layer)"
    array = np.asarray(raw_cam, dtype=np.float64)
    if array.size == 0:
        return "invalid", float("nan"), "CAM rỗng"
    bad = int(np.count_nonzero(~np.isfinite(array)))
    if bad:
        return "invalid", float("nan"), f"CAM có {bad} giá trị NaN/Inf"
    maximum = float(max(array.max(), 0.0))
    if maximum <= ZERO_TOLERANCE:
        return "all_zero", maximum, (
            "CAM dương bằng 0 trên toàn volume: theo phương pháp này không vùng nào đẩy logit lên"
        )
    return "ok", maximum, ""


def slice_scores(cam_positive: np.ndarray) -> np.ndarray:
    """Mean positive CAM of every axial slice (axis 0), in raw CAM units."""
    array = np.asarray(cam_positive, dtype=np.float64)
    return np.clip(array, 0.0, None).mean(axis=(1, 2))


def rank_slices(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(competition rank per slice, slice order): rank 1 = highest score; ties share a rank
    and are ordered by slice index."""
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(values)), -values))
    ranks = np.array([1 + int(np.count_nonzero(values > value)) for value in values], dtype=int)
    return ranks, order


def top_slices(scores: np.ndarray, count: int = MONTAGE_SLICES) -> list[int]:
    """The ``count`` highest-scoring slices with a positive score, best first."""
    _, order = rank_slices(scores)
    return [int(z) for z in order[: int(count)] if scores[int(z)] > 0]


# ---------------------------------------------------------------------------------------
# Geometry: orientation labels and input slice -> original NIfTI slice
# ---------------------------------------------------------------------------------------


def display_orientation(orientation: str | None, affine: Any = None) -> tuple[dict[str, str] | None, str]:
    """Edge labels of an S,P,L axial display, or None when the metadata does not fix them.

    ``orientation`` is the array orientation recorded by preprocessing; when ``affine`` is
    given it must agree with it, otherwise the labels are withheld instead of guessed.
    """
    code = str(orientation or "").upper()
    pairs = ({"L", "R"}, {"A", "P"}, {"S", "I"})
    if len(code) != 3 or any(len(set(code) & pair) != 1 for pair in pairs):
        return None, "hướng ảnh không có trong metadata"
    if affine is not None:
        try:
            import nibabel as nib

            from_affine = "".join(nib.aff2axcodes(np.asarray(affine, dtype=float)))
        except Exception as exc:  # noqa: BLE001 - an unreadable affine means "unknown"
            return None, f"affine không đọc được ({type(exc).__name__})"
        if from_affine != code:
            return None, f"orientation={code} mâu thuẫn với affine ({from_affine})"
    return {"top": "A", "bottom": "P", "left": "R", "right": "L"}, (
        f"mảng {code} xoay về S,P,L: trước (A) ở trên, trái bệnh nhân (L) bên phải ảnh"
    )


def _zoom_source(index: float, source_length: int, target_length: int) -> float:
    """Source coordinate of target voxel ``index`` for ``scipy.ndimage.zoom`` (grid_mode=False
    maps the first and last voxel centres onto each other)."""
    if target_length == source_length:
        return float(index)
    if target_length <= 1:
        return 0.0
    return float(index) * (source_length - 1) / (target_length - 1)


class _AxisChain:
    """Per-axis inverse of ``preprocess_volume_with_metadata`` read from its sidecar.

    Reorientation only permutes/flips axes and every later step (``ndimage.zoom``, body
    crop, canvas crop/pad or fit) acts on each axis independently, so an input index maps
    to an original index axis by axis.
    """

    def __init__(self, metadata: Mapping[str, Any]):
        import nibabel as nib

        self.orientation = str(metadata.get("orientation") or "").upper()
        self.original_affine = np.asarray(metadata["original_affine"], dtype=float)
        self.original_shape = [int(value) for value in metadata["original_shape"]][:3]
        self.output_shape = [int(value) for value in metadata["output_shape"]][-3:]
        self.resampled_shape = [int(value) for value in metadata["resampled_shape_before_crop"]]
        self.crop_box = [int(value) for value in metadata["crop_box"]]
        self.strategy = str(metadata.get("spatial_strategy") or "centre_crop_or_pad")
        self.source_bounds = [[int(value) for value in pair] for pair in metadata["source_bounds"]]
        self.target_bounds = [[int(value) for value in pair] for pair in metadata["target_bounds"]]
        self.cropped_shape = [int(value) for value in metadata["cropped_shape_before_fit"]]
        self.output_affine = np.asarray(metadata["output_affine"], dtype=float)
        pairs = ({"L", "R"}, {"A", "P"}, {"S", "I"})
        if len(self.orientation) != 3 or any(len(set(self.orientation) & pair) != 1 for pair in pairs):
            raise ValueError("orientation không xác định")
        if self.strategy not in {"centre_crop_or_pad", "fit"}:
            raise ValueError(f"spatial_strategy={self.strategy} chưa hỗ trợ")
        self.transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(self.original_affine),
            nib.orientations.axcodes2ornt(tuple(self.orientation)),
        )

    def source_axis(self, axis: int) -> int:
        return next(index for index, (target, _) in enumerate(self.transform) if int(target) == axis)

    def original_position(self, axis: int, index: float) -> tuple[str, float]:
        """("mapped", continuous original index) of input ``index`` on input ``axis``, or
        ("padding", nan) for canvas padding, ("outside", nan) beyond the original array."""
        if self.strategy == "fit":
            cropped = _zoom_source(index, self.cropped_shape[axis], self.output_shape[axis])
        else:
            start, stop = self.target_bounds[axis]
            if not start <= index < stop:
                return "padding", float("nan")
            cropped = float(index) - start + self.source_bounds[axis][0]
        resampled = cropped + self.crop_box[2 * axis]
        source_axis = self.source_axis(axis)
        length = self.original_shape[source_axis]
        oriented = _zoom_source(resampled, length, self.resampled_shape[axis])
        position = (length - 1 - oriented) if float(self.transform[source_axis][1]) < 0 else oriented
        if not -1e-6 <= position <= length - 1 + 1e-6:
            return "outside", float("nan")
        return "mapped", float(position)


def input_to_original(metadata: Mapping[str, Any], axis: int, indices: Sequence[float]) -> tuple[int, np.ndarray]:
    """(original axis, continuous original indices; NaN for padding/outside) of input
    ``indices`` along input ``axis`` (vectorised form of the chain used for the slice map)."""
    chain = _AxisChain(metadata)
    positions = np.array([chain.original_position(axis, value)[1] for value in indices], dtype=float)
    return chain.source_axis(axis), positions


def original_slice_mapping(metadata: Mapping[str, Any] | None, depth: int) -> dict[str, Any]:
    """Map every display slice z (S axis, 0 = most inferior) to the original NIfTI slice.

    The chain follows ``preprocess_volume_with_metadata`` step by step - canvas crop/pad or
    fit, body crop, ``ndimage.zoom`` resampling, axis reorientation - using the recorded
    shapes and bounds, so it reflects which original slices the data was interpolated from.
    The sidecar ``output_affine`` assumes a pure spacing scale for the zoom; the difference
    between both is returned as ``affine_discrepancy_mm``.
    """

    def unavailable(reason: str) -> dict[str, Any]:
        return {"available": False, "reason": reason, "entries": []}

    if not metadata:
        return unavailable("không có sidecar tiền xử lý")
    try:
        chain = _AxisChain(metadata)
    except (KeyError, TypeError) as exc:
        return unavailable(f"sidecar thiếu trường {exc}")
    except ValueError as exc:
        return unavailable(str(exc))
    orientation = chain.orientation
    axis = next(index for index, code in enumerate(orientation) if code in "SI")
    if chain.output_shape[axis] != int(depth):
        return unavailable(f"số slice hiển thị {depth} khác output_shape {chain.output_shape}")
    source_axis = chain.source_axis(axis)
    source_length = chain.original_shape[source_axis]
    original_affine, output_affine = chain.original_affine, chain.output_affine
    direction = output_affine[:3, axis]
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)

    entries: list[dict[str, Any]] = []
    discrepancy = 0.0
    for z in range(int(depth)):
        index = z if orientation[axis] == "S" else int(depth) - 1 - z
        status, position = chain.original_position(axis, index)
        if status != "mapped":
            entries.append({"z": z, "status": status})
            continue
        lower = math.floor(position + 1e-9)
        upper = min(lower + 1, source_length - 1) if position - lower > 1e-6 else lower
        world_output = float(unit @ (output_affine[:3, axis] * index + output_affine[:3, 3]))
        world_original = float(
            unit @ (original_affine[:3, source_axis] * position + original_affine[:3, 3])
        )
        discrepancy = max(discrepancy, abs(world_output - world_original))
        entries.append(
            {"z": z, "status": "mapped", "position": position, "lower": lower, "upper": upper}
        )
    return {
        "available": True,
        "reason": "",
        "axis": int(source_axis),
        "axis_name": "ijk"[source_axis],
        "length": int(source_length),
        "original_spacing_mm": float(np.linalg.norm(original_affine[:3, source_axis])),
        "entries": entries,
        "affine_discrepancy_mm": discrepancy,
    }


def padding_fraction(cam_positive: np.ndarray, metadata: Mapping[str, Any] | None) -> float | None:
    """Share of the positive CAM mass in canvas padding (outside the recorded target box).

    ``cam_positive`` must be on the input grid in input-array order. None when the sidecar
    does not describe a crop/pad canvas of this shape.
    """
    if not metadata or str(metadata.get("spatial_strategy") or "") != "centre_crop_or_pad":
        return None
    try:
        bounds = [[int(value) for value in pair] for pair in metadata["target_bounds"]]
        shape = [int(value) for value in metadata["output_shape"]][-3:]
    except (KeyError, TypeError, ValueError):
        return None
    array = np.asarray(cam_positive, dtype=np.float64)
    total = float(array.sum())
    if list(array.shape) != shape or not math.isfinite(total) or total <= 0:
        return None
    inside = array[tuple(slice(start, stop) for start, stop in bounds)].sum()
    return float((total - inside) / total)


def outside_body_fraction(hu: np.ndarray, cam_positive: np.ndarray) -> float | None:
    """Share of the positive CAM mass outside a rough body mask (QC number only).

    Body = largest 3-D component above -500 HU with every axial slice hole-filled. The mask
    is used for this number only; the overlay is never masked.
    """
    from scipy import ndimage

    total = float(np.asarray(cam_positive, dtype=np.float64).sum())
    if not math.isfinite(total) or total <= 0:
        return None
    labels, count = ndimage.label(np.asarray(hu) > -500.0)
    if count == 0:
        return 1.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    body = labels == int(np.argmax(sizes))
    del labels
    for z in range(body.shape[0]):
        body[z] = ndimage.binary_fill_holes(body[z])
    return float(np.asarray(cam_positive, dtype=np.float64)[~body].sum() / total)


# ---------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------


def format_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    if number != 0 and (abs(number) < 1e-3 or abs(number) >= 1e4):
        return f"{number:.{max(1, digits - 1)}e}"
    return f"{number:.{digits}g}"


def display_rows(vmax: float, hu_known: bool, interpolation: str) -> list[tuple[str, str]]:
    """Technical rows describing how every preview draws and scores the CAM."""
    interpolation_text = (
        f"{interpolation}; dùng cho điểm slice, montage và hiển thị mặc định; 'ô feature gốc' = "
        "nearest. Nội suy chỉ làm mượt, không tăng độ chính xác định vị"
    )
    normalisation_text = (
        f"CAM dương / max trên lưới feature của cả volume (max = {format_number(vmax)}); một "
        "colormap + colorbar cho mọi slice và montage; không chuẩn hóa từng slice. CAM toàn 0 "
        "hoặc NaN/Inf: không vẽ, không xếp hạng"
    )
    score_text = (
        "trung bình CAM dương của slice (đơn vị thô, trước chuẩn hóa hiển thị); montage = "
        f"{MONTAGE_SLICES} slice điểm cao nhất (hòa → slice đầu vào nhỏ hơn trước)"
    )
    opacity_text = (
        "overlay alpha tại mỗi pixel = opacity × CAM dương đã chuẩn hóa; CAM = 0 giữ nguyên CT, "
        "không ngưỡng hay mask; opacity mặc định "
        f"{DEFAULT_OPACITY:g}, chỉnh bằng thanh trượt"
    )
    window_text = (
        f"W/L {DEFAULT_WINDOW[0]:g}/{DEFAULT_WINDOW[1]:g} HU"
        if hu_known else "percentile 1–99 (HU không xác định từ metadata)"
    )
    return [
        ("Nội suy hiển thị", interpolation_text),
        ("Chuẩn hóa CAM", normalisation_text),
        ("Điểm slice", score_text),
        ("Opacity", opacity_text),
        ("CT window", window_text),
    ]


def contribution_text(raw_cam: Any) -> str:
    """First-order positive/negative CAM totals before the ReLU."""
    if raw_cam is None:
        return "N/A"
    array = np.asarray(raw_cam, dtype=np.float64)
    if not np.isfinite(array).all():
        return "N/A (CAM không hữu hạn)"
    positive = format_number(np.clip(array, 0, None).sum())
    negative = format_number(np.clip(array, None, 0).sum())
    return f"Σ dương = {positive}, Σ âm = {negative}; ReLU bỏ phần âm (vùng kéo logit xuống)"


def _slice_texts(case: Mapping[str, Any]) -> list[dict[str, str]]:
    depth = int(case["shape"][0])
    scores = case.get("scores")
    ranks = case.get("ranks")
    mapping = case.get("mapping") or {}
    entries = {int(entry["z"]): entry for entry in mapping.get("entries") or ()}
    # Models whose input is not an inferior-first axial stack (PENet windows) name their
    # own slices; otherwise slice z is counted from the inferior end.
    custom = list(case.get("slice_labels") or ())
    texts = []
    for z in range(depth):
        entry = entries.get(z)
        if not mapping.get("available"):
            original = f"N/A ({mapping.get('reason') or 'không xác định được'})"
            short = "N/A"
        elif entry is None or entry.get("status") == "outside":
            original = short = "N/A (ngoài phạm vi ảnh gốc)"
        elif entry["status"] == "padding":
            original = "N/A — padding của canvas (không có dữ liệu quét)"
            short = "padding, không có dữ liệu quét"
        else:
            position = float(entry["position"])
            source = (
                f"#{entry['lower']}" if entry["lower"] == entry["upper"]
                else f"nội suy #{entry['lower']}/#{entry['upper']}"
            )
            original = (
                f"{position:.1f} ({source}) / {mapping['length']} — trục {mapping['axis_name']} "
                "của NIfTI gốc, đánh số từ 0"
            )
            short = f"{position:.1f} ({source}) / {mapping['length']}"
        if scores is None or case.get("status") != "ok":
            score = "N/A"
            rank = "N/A"
        else:
            score = format_number(scores[z])
            rank = f"{int(ranks[z])}/{depth}"
            if float(scores[z]) <= 0:
                rank += " (đồng hạng, điểm 0)"
        label = custom[z] if z < len(custom) else {}
        texts.append(
            {
                "input": label.get("input") or f"{z + 1}/{depth} (z = {z}, 0 = phía dưới/inferior)",
                "slice": label.get("slice") or f"{z + 1}/{depth}",
                "original": original,
                "original_short": short,
                "score": score,
                "rank": rank,
            }
        )
    return texts


def _ct_to_uint8(hu: np.ndarray, hu_known: bool, window: tuple[float, float]) -> tuple[np.ndarray, str]:
    array = np.asarray(hu, dtype=np.float32)
    if hu_known:
        width, level = (float(value) for value in window)
        lower, upper = level - width / 2.0, level + width / 2.0
        text = f"W/L = {width:g}/{level:g} HU [{lower:g}, {upper:g}]"
    else:
        lower, upper = (float(value) for value in np.percentile(array, [1, 99]))
        if not upper > lower:
            lower, upper = float(array.min()), float(array.max()) or 1.0
        text = f"percentile 1–99 của volume (HU không khôi phục được từ metadata): [{lower:.3g}, {upper:.3g}]"
    scaled = np.clip((array - lower) / max(upper - lower, 1e-12), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8), text


def _colormap_lut() -> np.ndarray:
    import matplotlib

    colormap = matplotlib.colormaps[COLORMAP]
    return np.rint(colormap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)


def _png_base64(array: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="PNG", compress_level=6)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------


def write_cam_preview(stem: Path, case: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write ``<stem>.html`` and ``<stem>.png`` for one study (see the module docstring).

    ``case`` keys: ``ct`` [Z,H,W] display array (HU when ``hu_known``), ``cam_up`` [Z,H,W]
    raw positive CAM on the input grid or None, ``cam_grid`` raw positive CAM on the feature
    grid or None, ``status``/``status_reason``/``vmax`` from :func:`cam_status`, ``scores``,
    ``ranks``, ``top``, ``mapping``, ``orientation`` (edge labels or None),
    ``orientation_text``, ``spacing`` (z, y, x mm or None), ``header`` and ``technical``
    (lists of (label, value)), ``warnings`` and ``notes`` (lists of str), ``title``.
    Optional: ``slice_labels`` (per slice {"input", "slice"} text) and ``z_blocks`` (number of
    independent equal z blocks of the feature grid, e.g. PENet windows; trilinear display
    never interpolates across a block boundary).
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    window = tuple(case.get("window") or DEFAULT_WINDOW)
    ct_u8, window_text = _ct_to_uint8(case["ct"], bool(case.get("hu_known")), window)  # type: ignore[arg-type]
    case = {**case, "shape": list(ct_u8.shape), "window_text": window_text}
    html_path = stem.with_suffix(".html")
    png_path = stem.with_suffix(".png")
    _write_html(html_path, case, ct_u8)
    _write_png(png_path, case, ct_u8)
    return html_path, png_path


def _write_png(destination: Path, case: Mapping[str, Any], ct_u8: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    spacing = case.get("spacing")
    aspect = float(spacing[1] / spacing[2]) if spacing else 1.0
    status = str(case.get("status"))
    top = list(case.get("top") or ())
    cam_up = case.get("cam_up")
    vmax = float(case.get("vmax") or 0.0)
    texts = _slice_texts(case)
    labels = case.get("orientation")

    import textwrap

    header = [
        textwrap.fill(f"{label}: {value}", width=78, subsequent_indent="      ")
        for label, value in case.get("header") or ()
    ]
    half = math.ceil(len(header) / 2)
    blocks = ("\n".join(header[:half]), "\n".join(header[half:]))
    header_lines = max(block.count("\n") + 1 for block in blocks)
    warning_text = "\n".join(
        [textwrap.fill("⚠ " + text, width=160) for text in case.get("warnings") or ()]
        + [textwrap.fill("• " + text, width=160) for text in case.get("notes") or ()]
    )
    warning_lines = warning_text.count("\n") + 1 if warning_text else 0
    # Layout in inches: title, two header columns, warnings, montage rows, footer.
    line = 0.21
    # + room for the two-line titles of the first montage row
    header_height = 0.55 + line * (header_lines + warning_lines) + 0.65
    columns, pair_rows = 4, (math.ceil(len(top) / 2) if status == "ok" and top else 1)
    row_height, footer_height = 3.9, 1.5
    total = header_height + row_height * pair_rows + footer_height
    figure = plt.figure(figsize=(16, total))
    figure.text(
        0.015, 1 - 0.12 / total, str(case.get("title") or "Grad-CAM preview"),
        ha="left", va="top", fontsize=13, weight="bold",
    )
    for column, block in enumerate(blocks):
        figure.text(
            0.015 + 0.5 * column, 1 - 0.5 / total, block, ha="left", va="top", fontsize=10,
            linespacing=1.45,
        )
    if warning_text:
        figure.text(
            0.015, 1 - (0.5 + line * header_lines + 0.12) / total, warning_text, ha="left",
            va="top", fontsize=10, color="#5d4037", linespacing=1.45,
        )
    grid = figure.add_gridspec(
        pair_rows + 1, columns, height_ratios=[row_height] * pair_rows + [footer_height],
        left=0.02, right=0.98, top=1 - header_height / total, bottom=0.01, hspace=0.35, wspace=0.05,
    )

    def draw_ct(axis: Any, z: int) -> None:
        axis.imshow(ct_u8[z], cmap="gray", vmin=0, vmax=255, origin="upper", aspect=aspect)
        axis.set_xticks([])
        axis.set_yticks([])
        if labels:
            for key, (x, y, ha, va) in {
                "top": (0.5, 0.99, "center", "top"), "bottom": (0.5, 0.01, "center", "bottom"),
                "left": (0.01, 0.5, "left", "center"), "right": (0.99, 0.5, "right", "center"),
            }.items():
                axis.text(x, y, labels[key], color="white", fontsize=8, ha=ha, va=va,
                          transform=axis.transAxes, weight="bold")

    if status == "ok" and top:
        for order, z in enumerate(top):
            row, column = order // 2, (order % 2) * 2
            ct_axis = figure.add_subplot(grid[row, column])
            cam_axis = figure.add_subplot(grid[row, column + 1])
            draw_ct(ct_axis, z)
            draw_ct(cam_axis, z)
            cam_axis.imshow(
                np.asarray(cam_up[z]) / vmax, cmap=COLORMAP, vmin=0.0, vmax=1.0,
                alpha=DEFAULT_OPACITY * np.clip(np.asarray(cam_up[z]) / vmax, 0.0, 1.0),
                origin="upper", aspect=aspect, interpolation="nearest",
            )
            ct_axis.set_title(
                f"#{order + 1} · slice {texts[z]['slice']} · CT\n"
                f"gốc: {texts[z]['original_short']}", fontsize=8.5,
            )
            cam_axis.set_title(
                f"CT + Grad-CAM · điểm {texts[z]['score']} · hạng {texts[z]['rank']}", fontsize=8.5,
            )
    else:
        axis = figure.add_subplot(grid[0, :])
        axis.axis("off")
        message = str(case.get("status_reason") or "không có slice nào có điểm CAM dương")
        axis.text(0.5, 0.5, f"Không có montage: {message}", ha="center", va="center", fontsize=13,
                  color="#b71c1c", transform=axis.transAxes)

    footer = figure.add_subplot(grid[-1, :])
    footer.axis("off")
    if status == "ok":
        bar_axis = footer.inset_axes((0.0, 0.62, 0.45, 0.22))
        bar = ScalarMappable(norm=Normalize(0.0, vmax), cmap=COLORMAP)
        colorbar = figure.colorbar(bar, cax=bar_axis, orientation="horizontal")
        colorbar.set_label(
            "Grad-CAM dương (đơn vị thô) — một thang cho toàn volume, 0 → max = "
            f"{format_number(vmax)}; overlay opacity {DEFAULT_OPACITY:g}", fontsize=9,
        )
        colorbar.ax.tick_params(labelsize=8)
    footer_text = "\n".join(
        textwrap.fill(text, width=100 if status == "ok" else 190)
        for text in (
            MONTAGE_NOTE,
            DISCLAIMER,
            f"CT: {case.get('window_text')}. Hướng: {case.get('orientation_text')}.",
            "Viewer đầy đủ (mọi slice, thanh trượt, opacity, mục kỹ thuật): file .html cùng tên.",
        )
    )
    footer.text(
        0.47 if status == "ok" else 0.0, 0.98, footer_text,
        ha="left", va="top", fontsize=9, transform=footer.transAxes, linespacing=1.35,
    )
    temporary = destination.with_name(f".{destination.stem}.tmp.png")
    figure.savefig(temporary, dpi=80)
    plt.close(figure)
    temporary.replace(destination)


_STYLE = """
:root{color-scheme:dark;--bg:#0f1318;--panel:#171d25;--line:#2a3441;--text:#e6ebf1;
--muted:#9aa7b6;--warn:#ffb4a8;--warn-bg:#3a1d1a;--accent:#8ab4f8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 10px}
h2{font-size:16px;margin:0 0 8px}
section,details{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 14px}
.fields{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:6px 18px;margin:0}
.fields div{display:flex;gap:8px;min-width:0}
.fields div.wide{grid-column:1/-1}
.fields dt{color:var(--muted);flex:0 0 auto}
.fields dd{margin:0;font-weight:600;overflow-wrap:anywhere}
.note{color:var(--muted);font-size:13px;margin:8px 0 0}
.warn{background:var(--warn-bg);color:var(--warn);border-radius:8px;padding:8px 10px;margin:8px 0 0}
.info-note{background:#1d2632;border-radius:8px;padding:8px 10px;margin:8px 0 0}
.controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin-bottom:10px}
.controls label{display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--accent)}
#slice{flex:1 1 320px;min-width:200px}
button,select{background:#222b36;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font:inherit}
button:hover{border-color:var(--accent)}
.info{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:4px 18px;margin:0 0 10px}
.info dt{color:var(--muted)} .info dd{margin:0 0 4px;font-weight:600;overflow-wrap:anywhere}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
figure{margin:0;min-width:0}
figcaption{color:var(--muted);font-size:13px;margin-bottom:4px}
.frame{position:relative;width:100%}
.frame canvas{display:block;width:100%;height:auto;background:#000}
.edge{position:absolute;color:#fff;font:700 12px system-ui;text-shadow:0 0 3px #000;pointer-events:none}
.edge.t{top:4px;left:50%;transform:translateX(-50%)} .edge.b{bottom:4px;left:50%;transform:translateX(-50%)}
.edge.l{left:5px;top:50%;transform:translateY(-50%)} .edge.r{right:5px;top:50%;transform:translateY(-50%)}
.edge.small{font-size:10px}
.blocked{position:absolute;inset:auto 8px 8px 8px;background:rgba(58,29,26,.9);color:var(--warn);padding:6px 8px;border-radius:6px;font-size:13px}
.colorbar{display:grid;grid-template-columns:minmax(160px,420px) 1fr;gap:6px 14px;align-items:center;margin-top:12px}
.colorbar canvas{width:100%;height:14px;border:1px solid var(--line);border-radius:3px}
.ticks{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
.montage{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.tile{border:1px solid var(--line);border-radius:8px;padding:6px;cursor:pointer;background:#11161d}
.tile:hover,.tile.active{border-color:var(--accent)}
.tile .pair{gap:4px}
.tile p{margin:4px 0 0;font-size:12px;color:var(--muted)}
details summary{cursor:pointer;font-weight:600}
.tech{display:grid;grid-template-columns:minmax(160px,260px) 1fr;gap:6px 16px;margin:10px 0 0}
.tech dt{color:var(--muted)} .tech dd{margin:0;overflow-wrap:anywhere}
@media (max-width:640px){.pair{grid-template-columns:1fr}.tech{grid-template-columns:1fr}.colorbar{grid-template-columns:1fr}}
"""

_SCRIPT = r"""
(() => {
"use strict";
const D = JSON.parse(document.getElementById("cam-data").textContent);
const [Z, H, W] = D.shape;
const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
const LUT = bytes(D.lut);
const CAM = D.cam ? new Float32Array(bytes(D.cam).buffer) : null;
const $ = (id) => document.getElementById(id);
const slider = $("slice"), opacity = $("opacity"), mode = $("mode");
const ctCanvas = $("ct"), camCanvas = $("cam");
for (const c of [ctCanvas, camCanvas]) { c.width = W; c.height = H; }

// torch.nn.functional.interpolate(..., align_corners=False) source index for one axis.
function source(dst, inN, outN) {
  let s = (dst + 0.5) * (inN / outN) - 0.5;
  if (s < 0) s = 0;
  const i0 = Math.floor(s);
  return [i0, i0 < inN - 1 ? i0 + 1 : i0, s - i0];
}
const planes = new Map();
function camPlane(z, how) {
  const key = how + ":" + z;
  if (planes.has(key)) return planes.get(key);
  const [gz, gy, gx] = D.grid;
  const out = new Float32Array(H * W);
  if (how === "nearest") {
    const iz = Math.min(gz - 1, Math.floor(z * gz / Z));
    for (let y = 0; y < H; y++) {
      const row = (iz * gy + Math.min(gy - 1, Math.floor(y * gy / H))) * gx;
      for (let x = 0; x < W; x++) out[y * W + x] = CAM[row + Math.min(gx - 1, Math.floor(x * gx / W))];
    }
  } else {
    // Independent z blocks (PENet windows) are interpolated inside their own block only.
    const zb = Z / D.zblocks, gb = gz / D.zblocks, block = Math.floor(z / zb);
    const [b0, b1, lz] = source(z - block * zb, gb, zb);
    const z0 = block * gb + b0, z1 = block * gb + b1;
    const plane = new Float32Array(gy * gx);
    for (let i = 0; i < gy * gx; i++) plane[i] = (1 - lz) * CAM[z0 * gy * gx + i] + lz * CAM[z1 * gy * gx + i];
    const xs = Array.from({ length: W }, (_, x) => source(x, gx, W));
    for (let y = 0; y < H; y++) {
      const [y0, y1, ly] = source(y, gy, H);
      for (let x = 0; x < W; x++) {
        const [x0, x1, lx] = xs[x];
        const a = plane[y0 * gx + x0] * (1 - lx) + plane[y0 * gx + x1] * lx;
        const b = plane[y1 * gx + x0] * (1 - lx) + plane[y1 * gx + x1] * lx;
        out[y * W + x] = a * (1 - ly) + b * ly;
      }
    }
  }
  planes.set(key, out);
  if (planes.size > 48) planes.delete(planes.keys().next().value);
  return out;
}
const gray = new Array(Z).fill(null);
function ctSlice(z) {
  if (!gray[z]) {
    gray[z] = new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        const c = document.createElement("canvas");
        c.width = W; c.height = H;
        const g = c.getContext("2d", { willReadFrequently: true });
        g.drawImage(image, 0, 0);
        const rgba = g.getImageData(0, 0, W, H).data;
        const values = new Uint8Array(W * H);
        for (let i = 0; i < W * H; i++) values[i] = rgba[4 * i];
        resolve(values);
      };
      image.onerror = () => reject(new Error("CT slice " + z + " could not be decoded"));
      image.src = "data:image/png;base64," + D.ct[z];
    });
  }
  return gray[z];
}
function paint(canvas, values, plane, alpha) {
  const g = canvas.getContext("2d");
  const image = g.createImageData(W, H);
  const o = image.data;
  for (let i = 0, p = 0; i < W * H; i++, p += 4) {
    const v = values[i];
    if (plane) {
      const k = 3 * Math.max(0, Math.min(255, Math.round(plane[i] * 255)));
      const strength = Math.max(0, Math.min(1, plane[i]));
      const a = alpha * strength;
      o[p] = (1 - a) * v + a * LUT[k];
      o[p + 1] = (1 - a) * v + a * LUT[k + 1];
      o[p + 2] = (1 - a) * v + a * LUT[k + 2];
    } else {
      o[p] = o[p + 1] = o[p + 2] = v;
    }
    o[p + 3] = 255;
  }
  g.putImageData(image, 0, 0);
}
let token = 0;
async function show(z) {
  z = Math.max(0, Math.min(Z - 1, z | 0));
  slider.value = String(z);
  const t = D.slices[z];
  $("i-input").textContent = t.input;
  $("i-original").textContent = t.original;
  $("i-score").textContent = t.score;
  $("i-rank").textContent = t.rank;
  for (const tile of document.querySelectorAll(".tile")) tile.classList.toggle("active", Number(tile.dataset.z) === z);
  const mine = ++token;
  const values = await ctSlice(z);
  if (mine !== token) return;
  paint(ctCanvas, values, null, 0);
  paint(camCanvas, values, CAM ? camPlane(z, mode.value) : null, Number(opacity.value));
  for (const n of [z - 1, z + 1]) if (n >= 0 && n < Z) ctSlice(n);
}
async function paintMontage() {
  for (const tile of document.querySelectorAll(".tile")) {
    const z = Number(tile.dataset.z);
    const [a, b] = tile.querySelectorAll("canvas");
    for (const c of [a, b]) { c.width = W; c.height = H; }
    const values = await ctSlice(z);
    paint(a, values, null, 0);
    paint(b, values, camPlane(z, mode.value), Number(opacity.value));
  }
}
function refresh() {
  $("opacity-value").textContent = Number(opacity.value).toFixed(2);
  show(Number(slider.value));
  if (CAM) paintMontage();
}
const bar = $("bar");
if (bar) {
  bar.width = 256; bar.height = 1;
  const g = bar.getContext("2d"), image = g.createImageData(256, 1);
  for (let i = 0; i < 256; i++) {
    image.data.set([LUT[3 * i], LUT[3 * i + 1], LUT[3 * i + 2], 255], 4 * i);
  }
  g.putImageData(image, 0, 0);
}
slider.addEventListener("input", () => show(Number(slider.value)));
opacity.addEventListener("input", refresh);
mode.addEventListener("change", refresh);
$("prev").addEventListener("click", () => show(Number(slider.value) - 1));
$("next").addEventListener("click", () => show(Number(slider.value) + 1));
$("best").addEventListener("click", () => show(D.top.length ? D.top[0] : D.start));
for (const tile of document.querySelectorAll(".tile")) {
  tile.addEventListener("click", () => { show(Number(tile.dataset.z)); ctCanvas.scrollIntoView({ block: "nearest" }); });
}
document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowDown") show(Number(slider.value) - 1);
  if (event.key === "ArrowRight" || event.key === "ArrowUp") show(Number(slider.value) + 1);
});
slider.value = String(D.start);
refresh();
})();
"""


def _definition_list(items: Sequence[tuple[str, Any]], css_class: str) -> str:
    rows = []
    for label, value in items:
        term, text = html.escape(str(label)), html.escape(str(value))
        if css_class == "fields":
            # Long values (e.g. the threshold rule) get a full row instead of a narrow column.
            wide = ' class="wide"' if len(str(value)) > 45 else ""
            rows.append(f"<div{wide}><dt>{term}</dt><dd>{text}</dd></div>")
        else:
            rows.append(f"<dt>{term}</dt><dd>{text}</dd>")
    return f'<dl class="{css_class}">{"".join(rows)}</dl>'


def _write_html(destination: Path, case: Mapping[str, Any], ct_u8: np.ndarray) -> None:
    depth, height, width = ct_u8.shape
    status = str(case.get("status"))
    top = [int(z) for z in (case.get("top") or ())] if status == "ok" else []
    vmax = float(case.get("vmax") or 0.0)
    spacing = case.get("spacing")
    ratio = (width * float(spacing[2])) / (height * float(spacing[1])) if spacing else width / height
    grid = case.get("cam_grid")
    cam_payload = None
    if status == "ok" and grid is not None:
        normalised = np.ascontiguousarray(np.asarray(grid, dtype=np.float64) / vmax, dtype="<f4")
        cam_payload = base64.b64encode(normalised.tobytes()).decode("ascii")
    start = top[0] if top else depth // 2
    payload = {
        "shape": [int(depth), int(height), int(width)],
        "grid": [int(value) for value in np.shape(grid)] if cam_payload else None,
        "cam": cam_payload,
        "lut": base64.b64encode(_colormap_lut().tobytes()).decode("ascii"),
        "ct": [_png_base64(ct_u8[z]) for z in range(depth)],
        "slices": _slice_texts(case),
        "top": top,
        "start": int(start),
        "zblocks": int(case.get("z_blocks") or 1),
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    labels = case.get("orientation")

    def frame(canvas_id: str, blocked: str = "") -> str:
        edges = "".join(
            f'<span class="edge {css}">{html.escape(labels[key])}</span>'
            for key, css in (("top", "t"), ("bottom", "b"), ("left", "l"), ("right", "r"))
        ) if labels else ""
        overlay = f'<div class="blocked">{html.escape(blocked)}</div>' if blocked else ""
        return (
            f'<div class="frame" style="aspect-ratio:{ratio:.6f}">'
            f'<canvas id="{canvas_id}" style="aspect-ratio:{ratio:.6f}"></canvas>{edges}{overlay}</div>'
        )

    warnings = "".join(f'<p class="warn">{html.escape(text)}</p>' for text in case.get("warnings") or ())
    warnings += "".join(f'<p class="info-note">{html.escape(text)}</p>' for text in case.get("notes") or ())
    blocked = "" if status == "ok" else f"Không vẽ CAM: {case.get('status_reason')}"
    texts = payload["slices"]
    tile_edges = "".join(
        f'<span class="edge small {css}">{html.escape(labels[key])}</span>'
        for key, css in (("top", "t"), ("bottom", "b"), ("left", "l"), ("right", "r"))
    ) if labels else ""
    tile_frame = f'<div class="frame" style="aspect-ratio:{ratio:.6f}"><canvas></canvas>{tile_edges}</div>'
    tiles = "".join(
        f'<div class="tile" data-z="{z}"><div class="pair">{tile_frame}{tile_frame}</div>'
        f"<p>#{order + 1} · slice {html.escape(texts[z]['slice'])} · gốc "
        f"{html.escape(texts[z]['original_short'])} · điểm "
        f"{html.escape(texts[z]['score'])} · hạng {html.escape(texts[z]['rank'])}</p></div>"
        for order, z in enumerate(top)
    )
    if status == "ok" and top:
        montage_body = f'<div class="montage">{tiles}</div>'
    else:
        reason = case.get("status_reason") if status != "ok" else "không slice nào có điểm CAM dương"
        montage_body = f'<p class="warn">Không có montage: {html.escape(str(reason))}</p>'
    colorbar = (
        '<div class="colorbar"><div><canvas id="bar"></canvas><div class="ticks">'
        f"<span>0</span><span>{html.escape(format_number(vmax / 2))}</span>"
        f"<span>{html.escape(format_number(vmax))}</span></div></div>"
        '<p class="note">Grad-CAM dương, đơn vị thô. Một thang màu chung cho mọi slice và '
        f"montage: 0 → max của volume ({html.escape(format_number(vmax))}); không chuẩn hóa "
        f"riêng từng slice. Colormap {COLORMAP}; overlay alpha = opacity × CAM chuẩn hóa, CT gốc vẫn rõ khi CAM yếu, "
        "áp dụng đều mọi pixel, không ngưỡng, không mask cơ thể.</p></div>"
        if status == "ok" else ""
    )
    title = html.escape(str(case.get("title") or "Grad-CAM preview"))
    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{_STYLE}</style></head>
<body><main>
<section><h1>{title}</h1>{_definition_list(case.get("header") or (), "fields")}{warnings}
<p class="note">{html.escape(DISCLAIMER)}</p></section>
<section>
<div class="controls">
<button id="prev" type="button" aria-label="slice trước">◀</button>
<input id="slice" type="range" min="0" max="{depth - 1}" step="1" value="{start}" aria-label="slice đầu vào">
<button id="next" type="button" aria-label="slice sau">▶</button>
<button id="best" type="button">Slice CAM cao nhất</button>
<label>Opacity <input id="opacity" type="range" min="0" max="1" step="0.05" value="{DEFAULT_OPACITY}"><span id="opacity-value"></span></label>
<label>Hiển thị CAM <select id="mode"><option value="trilinear">nội suy trilinear (mặc định)</option><option value="nearest">ô feature gốc (nearest)</option></select></label>
</div>
<dl class="info">
<div><dt>Slice đầu vào</dt><dd id="i-input"></dd></div>
<div><dt>Slice gốc tương ứng</dt><dd id="i-original"></dd></div>
<div><dt>Điểm CAM (mean CAM dương)</dt><dd id="i-score"></dd></div>
<div><dt>Hạng trong volume</dt><dd id="i-rank"></dd></div>
<div><dt>Hướng</dt><dd>{html.escape(str(case.get("orientation_text") or "N/A"))}</dd></div>
<div><dt>Cửa sổ CT</dt><dd>{html.escape(str(case.get("window_text")))}</dd></div>
</dl>
<div class="pair">
<figure><figcaption>{html.escape(str(case.get("ct_caption") or "CT trên lưới đầu vào"))}</figcaption>{frame("ct")}</figure>
<figure><figcaption>CT + Grad-CAM</figcaption>{frame("cam", blocked)}</figure>
</div>
{colorbar}
<p class="note">Nội suy trilinear chỉ làm mượt hiển thị; độ phân giải thật của CAM là một ô feature
(xem mục kỹ thuật). Chọn "ô feature gốc" để thấy lưới thật. Phím ←/→ đổi slice.</p>
</section>
<section><h2>Montage {MONTAGE_SLICES} slice có điểm CAM cao nhất</h2>
<p class="warn">{html.escape(MONTAGE_NOTE)}</p>{montage_body}
<p class="note">Bấm một ô để mở slice đó ở viewer phía trên.</p></section>
<details><summary>Chi tiết kỹ thuật</summary>{_definition_list(case.get("technical") or (), "tech")}</details>
</main>
<script type="application/json" id="cam-data">{data}</script>
<script>{_SCRIPT}</script>
</body></html>
"""
    temporary = destination.with_name(f".{destination.stem}.tmp.html")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(destination)


__all__ = [
    "CAM_MEANING",
    "CAM_METHODS",
    "DEFAULT_OPACITY",
    "DEFAULT_WINDOW",
    "DISCLAIMER",
    "MONTAGE_NOTE",
    "MONTAGE_SLICES",
    "cam_from_target_layer",
    "cam_status",
    "contribution_text",
    "display_orientation",
    "display_rows",
    "format_number",
    "input_to_original",
    "original_slice_mapping",
    "outside_body_fraction",
    "padding_fraction",
    "rank_slices",
    "slice_scores",
    "top_slices",
    "write_cam_preview",
]
