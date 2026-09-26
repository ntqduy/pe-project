from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .nifti import load_image, load_nifti


def representative_slice_indices(mask: np.ndarray, maximum_slices: int = 3) -> tuple[int, ...]:
    """Choose largest-area and deterministic before/after-center axial slices."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"preview mask must be 3D, got shape={binary.shape}")
    if maximum_slices < 1:
        return ()
    areas = binary.sum(axis=(0, 1))
    occupied = np.flatnonzero(areas)
    if not len(occupied):
        return (int(binary.shape[2] // 2),)

    largest = int(np.argmax(areas))
    choices = [largest]
    if maximum_slices == 2:
        extent_middle = int(round((int(occupied[0]) + int(occupied[-1])) / 2))
        if extent_middle != largest and areas[extent_middle] > 0:
            choices.append(extent_middle)
        else:
            for candidate in np.argsort(areas)[::-1]:
                index = int(candidate)
                if areas[index] <= 0:
                    break
                if index not in choices:
                    choices.append(index)
                    break
    elif maximum_slices > 2:
        before = int(occupied[(len(occupied) - 1) // 4])
        after = int(occupied[(3 * (len(occupied) - 1)) // 4])
        for candidate in (before, after):
            if candidate not in choices:
                choices.append(candidate)
        if len(choices) < min(maximum_slices, len(occupied)):
            for candidate in np.argsort(areas)[::-1]:
                index = int(candidate)
                if areas[index] <= 0:
                    break
                if index not in choices:
                    choices.append(index)
                if len(choices) >= maximum_slices:
                    break
    return tuple(choices[:maximum_slices])


def _write_slice(
    volume: np.ndarray,
    binary_mask: np.ndarray,
    destination: Path,
    *,
    slice_index: int,
    study_id: str,
    anatomy: str,
    source: str,
    status: str,
    note: str = "",
    patient_id: str | None = None,
    voxel_count: int | None = None,
    physical_volume_mm3: float | None = None,
    overlay_color: str = "orange",
    window_width: float = 700.0,
    window_level: float = 100.0,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for preview generation") from exc

    image_slice = np.asarray(volume[:, :, slice_index], dtype=float).T
    mask_slice = binary_mask[:, :, slice_index].T
    if window_width <= 0:
        raise ValueError("preview window_width must be positive")
    lower = float(window_level - window_width / 2.0)
    upper = float(window_level + window_width / 2.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.imshow(image_slice, cmap="gray", origin="lower", vmin=lower, vmax=upper)
    overlay = np.ma.masked_where(~mask_slice, mask_slice)
    from matplotlib.colors import ListedColormap

    axis.imshow(
        overlay,
        cmap=ListedColormap([overlay_color]),
        alpha=0.32,
        origin="lower",
        vmin=0,
        vmax=1,
    )
    if mask_slice.any():
        axis.contour(mask_slice.astype(float), levels=[0.5], colors=[overlay_color], linewidths=0.9)
    identity = f"patient={patient_id} | " if patient_id else ""
    metrics = ""
    if voxel_count is not None:
        metrics += f" | voxels={voxel_count}"
    if physical_volume_mm3 is not None:
        metrics += f" | volume={physical_volume_mm3:.1f} mm^3"
    axis.set_title(
        f"{identity}study={study_id} | {anatomy} | z={slice_index}{note}\n"
        f"{source} | QC={status}{metrics} | W/L={window_width:g}/{window_level:g}"
    )
    axis.axis("off")
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    return destination


def write_overlay_previews(
    volume_path: Path,
    mask_path: Path,
    destination_directory: Path,
    *,
    study_id: str,
    anatomy: str,
    source: str,
    status: str,
    maximum_slices: int = 3,
    patient_id: str | None = None,
    voxel_count: int | None = None,
    physical_volume_mm3: float | None = None,
    overlay_color: str = "orange",
    window_width: float = 700.0,
    window_level: float = 100.0,
) -> tuple[Path, ...]:
    """Write one to three representative CTPA overlays without a large image dump."""
    volume, _ = load_nifti(volume_path)
    mask, _ = load_nifti(mask_path)
    binary = np.asarray(mask, dtype=bool)
    indices = representative_slice_indices(binary, maximum_slices)
    outputs = []
    for ordinal, slice_index in enumerate(indices, start=1):
        safe_name = "".join(character if character.isalnum() else "_" for character in anatomy)
        destination = destination_directory / (
            f"{safe_name}_axial_slice_{ordinal:03d}_z{slice_index:04d}.png"
        )
        outputs.append(
            _write_slice(
                volume,
                binary,
                destination,
                slice_index=slice_index,
                study_id=study_id,
                anatomy=anatomy,
                source=source,
                status=status,
                patient_id=patient_id,
                voxel_count=voxel_count,
                physical_volume_mm3=physical_volume_mm3,
                overlay_color=overlay_color,
                window_width=window_width,
                window_level=window_level,
            )
        )
    return tuple(outputs)


# Statuses that make a panel title red: the mask is missing, empty or failed QC.
_BAD_STATUSES = {"FAIL", "UNAVAILABLE"}
# Large masks that cover most of the field of view; drawn as contours, not filled, on the
# coronal overview so the organs underneath stay visible.
_COVERING_ANATOMIES = {"body", "body_wall"}
# Left out of the coronal overview: the LungMask cross-check and the lung subdivisions, which
# only repeat `lung` there; each still gets its own axial panel.
_COMPOSITE_SKIPPED = {"lung_lungmask", "lung_left", "lung_right"}


def _las_orientation(image: Any) -> tuple[Any, tuple[float, float, float]]:
    """Orientation transform to LAS and the matching voxel spacing (L, A, S)."""
    import nibabel as nib

    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("L", "A", "S")),
    )
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    spacing = [0.0, 0.0, 0.0]
    for source_axis, (target_axis, _) in enumerate(transform):
        spacing[int(target_axis)] = zooms[source_axis]
    return transform, (spacing[0], spacing[1], spacing[2])


def _as_las(array: np.ndarray, transform: Any) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.orientations.apply_orientation(np.asarray(array), transform))


def las_axial(volume_las: np.ndarray, z: int) -> np.ndarray:
    """Axial slice of a LAS volume for ``imshow(..., origin="lower")``.

    Rows run posterior->anterior (anterior at the top), columns run right->left, so the
    patient's left is on the image right (radiological convention).
    """
    return np.asarray(volume_las[:, :, int(z)]).T


def las_coronal(volume_las: np.ndarray, y: int) -> np.ndarray:
    """Coronal slice of a LAS volume for ``imshow(..., origin="lower")`` (superior at the top)."""
    return np.asarray(volume_las[:, int(y), :]).T


def write_segmentation_contact_sheet(
    volume_path: Path,
    items: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    study_id: str,
    patient_id: str,
    window_width: float = 700.0,
    window_level: float = 100.0,
    columns: int = 5,
    title_note: str = "",
    pe_present: bool | None = None,
) -> Path:
    """Write one PNG that shows every anatomy mask of a study for visual QC.

    ``items`` are manifest rows with ``anatomy``, ``mask_path`` (None when the mask was not
    produced), ``status``, ``reason``, ``volume_ml`` and optionally ``duplicate_of``. Each
    panel shows the axial slice with the largest mask area; the last panel is a coronal
    overview of all masks with the physical aspect ratio. The CT is read once and every
    volume is reoriented to LAS, so the display is correct for any on-disk orientation.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for preview generation") from exc
    if window_width <= 0:
        raise ValueError("preview window_width must be positive")
    lower = float(window_level - window_width / 2.0)
    upper = float(window_level + window_width / 2.0)

    image = load_image(volume_path)
    transform, spacing = _las_orientation(image)
    volume = _as_las(np.asarray(image.dataobj), transform)
    axial_aspect = spacing[1] / spacing[0]
    coronal_aspect = spacing[2] / spacing[0]

    def load_mask(path: Any) -> np.ndarray | None:
        if not path or not Path(str(path)).is_file():
            return None
        mask, mask_image = load_nifti(str(path))
        mask_transform, _ = _las_orientation(mask_image)
        las = _as_las(np.asarray(mask) > 0, mask_transform)
        return las if las.shape == volume.shape else None

    # Coronal overview plane: through the largest lung cross-section when available.
    coronal_y = volume.shape[1] // 2
    by_name = {str(item.get("anatomy")): item for item in items}
    lung = load_mask((by_name.get("lung") or {}).get("mask_path"))
    if lung is not None and lung.any():
        coronal_y = int(np.argmax(lung.sum(axis=(0, 2))))
    del lung

    panels: list[dict[str, Any]] = []
    coronal_layers: list[tuple[str, np.ndarray]] = []
    for item in items:
        anatomy = str(item.get("anatomy"))
        status = str(item.get("status") or "")
        panel: dict[str, Any] = {"anatomy": anatomy, "status": status, "item": item}
        mask = load_mask(item.get("mask_path"))
        if mask is None:
            panel["kind"] = "unavailable"
        else:
            areas = mask.sum(axis=(0, 1))
            if areas.any():
                z = int(np.argmax(areas))
                panel["kind"] = "mask"
            else:
                z = volume.shape[2] // 2
                panel["kind"] = "empty"
            panel["z"] = z
            panel["ct"] = las_axial(volume, z)
            panel["mask"] = las_axial(mask, z)
            coronal = las_coronal(mask, coronal_y)
            if coronal.any() and anatomy not in _COMPOSITE_SKIPPED and "_lobe_" not in anatomy:
                coronal_layers.append((anatomy, coronal))
        panels.append(panel)
        del mask

    total = len(panels) + 1
    columns = max(1, int(columns))
    rows = math.ceil(total / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.6 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for index, panel in enumerate(panels):
        axis = axes.ravel()[index]
        item = panel["item"]
        anatomy = panel["anatomy"]
        status = panel["status"]
        duplicate = str(item.get("duplicate_of") or "")
        colour = "red" if status in _BAD_STATUSES or panel["kind"] != "mask" else (
            "darkorange" if status == "SUSPICIOUS" or duplicate else "black"
        )
        if panel["kind"] == "unavailable":
            reason = str(item.get("reason") or "not generated")
            axis.set_facecolor("0.85")
            axis.text(
                0.5, 0.55, "UNAVAILABLE", color="red", ha="center", va="center",
                fontsize=12, weight="bold", transform=axis.transAxes,
            )
            axis.text(
                0.5, 0.35, reason[:80], color="0.2", ha="center", va="center",
                fontsize=7, wrap=True, transform=axis.transAxes,
            )
            axis.set_title(f"{anatomy}\n{status}", fontsize=9, color=colour)
            continue
        axis.imshow(panel["ct"], cmap="gray", origin="lower", vmin=lower, vmax=upper, aspect=axial_aspect)
        if panel["kind"] == "empty":
            axis.text(
                0.5, 0.5, "EMPTY", color="red", ha="center", va="center",
                fontsize=16, weight="bold", transform=axis.transAxes,
            )
        else:
            overlay = np.ma.masked_where(~panel["mask"], panel["mask"])
            axis.imshow(
                overlay, cmap=ListedColormap(["orange"]), alpha=0.35, origin="lower",
                vmin=0, vmax=1, aspect=axial_aspect,
            )
            axis.contour(panel["mask"].astype(float), levels=[0.5], colors=["orange"], linewidths=0.7)
        volume_ml = item.get("volume_ml")
        volume_text = f"{float(volume_ml):.1f} mL" if volume_ml not in (None, "") else "- mL"
        second = f"z={panel['z']} | {volume_text} | {status}"
        if duplicate:
            second += f"\n= {duplicate} (identical)"
        axis.set_title(f"{anatomy}\n{second}", fontsize=8, color=colour)

    axis = axes.ravel()[len(panels)]
    axis.imshow(
        las_coronal(volume, coronal_y), cmap="gray", origin="lower",
        vmin=lower, vmax=upper, aspect=coronal_aspect,
    )
    palette = plt.get_cmap("tab20")
    filled = [(name, layer) for name, layer in coronal_layers if name not in _COVERING_ANATOMIES]
    for order, (_name, layer) in enumerate(filled):
        colour = palette(order % 20)
        axis.imshow(
            np.ma.masked_where(~layer, layer), cmap=ListedColormap([colour]), alpha=0.45,
            origin="lower", vmin=0, vmax=1, aspect=coronal_aspect,
        )
    for name, layer in coronal_layers:
        if name in _COVERING_ANATOMIES:
            axis.contour(layer.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.6)
    axis.set_title(f"coronal overview | y={coronal_y}\n(all masks, colour per anatomy)", fontsize=8)
    if filled:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette(order % 20)) for order in range(len(filled))
        ]
        axis.legend(
            handles, [name for name, _ in filled], loc="upper left", bbox_to_anchor=(1.0, 1.0),
            fontsize=6, frameon=False,
        )

    produced = sum(panel["kind"] == "mask" for panel in panels)
    pe_text = "Có" if pe_present is True else "Không" if pe_present is False else "Không rõ"
    figure.suptitle(
        f"patient={patient_id} | study={study_id} | PE: {pe_text} | "
        f"non-empty masks {produced}/{len(panels)} | "
        f"W/L={window_width:g}/{window_level:g} | axial: anterior up, patient left = image right | "
        f"z counted from the inferior end{(' | ' + title_note) if title_note else ''}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.png")
    figure.savefig(temporary, dpi=100)
    plt.close(figure)
    temporary.replace(destination)
    return destination
