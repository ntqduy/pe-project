from __future__ import annotations

from pathlib import Path

import numpy as np

from .nifti import load_nifti


def write_overlay_preview(
    volume_path: Path,
    mask_path: Path,
    destination: Path,
    *,
    study_id: str,
    anatomy: str,
    source: str,
    status: str,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for preview generation") from exc
    volume, _ = load_nifti(volume_path)
    mask, _ = load_nifti(mask_path)
    binary = mask.astype(bool)
    areas = binary.sum(axis=(0, 1))
    slice_index = int(np.argmax(areas)) if areas.any() else int(volume.shape[2] // 2)
    image_slice = np.asarray(volume[:, :, slice_index], dtype=float).T
    mask_slice = binary[:, :, slice_index].T
    lower, upper = np.percentile(image_slice, (1, 99))
    if upper <= lower:
        lower, upper = float(image_slice.min()), float(image_slice.max() + 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.imshow(image_slice, cmap="gray", origin="lower", vmin=lower, vmax=upper)
    overlay = np.ma.masked_where(~mask_slice, mask_slice)
    axis.imshow(overlay, cmap="autumn", alpha=0.4, origin="lower", vmin=0, vmax=1)
    axis.contour(mask_slice.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
    axis.set_title(f"{study_id} | {anatomy} | z={slice_index}\n{source} | QC={status}")
    axis.axis("off")
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    return destination
