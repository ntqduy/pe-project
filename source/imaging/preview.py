from __future__ import annotations

from pathlib import Path

import numpy as np

from .nifti import load_nifti


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
