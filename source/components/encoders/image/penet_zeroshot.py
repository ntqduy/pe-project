"""PENet applied unchanged, as a zero-shot PE classifier.

PENet is already a trained CTPA PE model, so running it on this project's cohort is a
genuine zero-shot baseline rather than another arm to train. Everything below is read off
``third_party/repos/penet`` rather than the paper, because the repository and the paper
disagree in two places that would silently change the number:

* ``ct/ct_pe_constants.py`` defines a second ``(-250, 450)`` window, but it sits inside a
  triple-quoted string and never executes. The live window is ``(-100, 900)``.
* ``datasets/base_ct_dataset._normalize_raw`` only subtracts the mean. ``CONTRAST_HU_STD``
  is defined but never applied, so dividing by it would be wrong.

PENet cannot read this project's ``volumes/*.npy`` cache: that cache is one channel,
resampled to 1.5 mm isotropic and fitted to 128^3, while PENet wants non-overlapping
32-slice windows of 208x208 in its own HU window. This module therefore reads the raw
NIfTI the manifest points at.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

# datasets/ct_pe_dataset_3d.py + args/base_arg_parser.py + ct/ct_pe_constants.py
AIR_HU_VAL = -1000.0
CONTRAST_HU_MIN = -100.0
CONTRAST_HU_MAX = 900.0
CONTRAST_HU_MEAN = 0.15897
NUM_SLICES = 32
RESIZE_SHAPE = (224, 224)
CROP_SHAPE = (208, 208)
PENET_WEIGHT_SHA256 = "891276e2fb085735f0cc39fab0bc9c0cd5ab0ae4ef9aa245e3b17b1d526bc52b"

PENET_CONTRACT: dict[str, Any] = {
    "slices_per_window": NUM_SLICES,
    "window_stride": NUM_SLICES,          # windows are non-overlapping
    "resize_shape": list(RESIZE_SHAPE),
    "crop_shape": list(CROP_SHAPE),
    "clip_hu": [CONTRAST_HU_MIN, CONTRAST_HU_MAX],
    "subtract_mean": CONTRAST_HU_MEAN,
    "divide_by_std": False,
    "pad_value_hu": AIR_HU_VAL,
    "input_channels": 1,                  # the model expands 1 -> 3 internally
    # DICOM pixel_array layout: rows anterior->posterior, columns patient right->left.
    "inplane_orientation": "rows_A_to_P__cols_R_to_L",
    "series_aggregation": "max_sigmoid_over_windows",
}


class PenetError(RuntimeError):
    pass


def load_penet(checkpoint: str | Path, repo: str | Path) -> tuple[Any, dict[str, Any]]:
    """Build PENetClassifier from the repository and strict-load the released weights."""
    import sys

    import torch

    checkpoint, repo = Path(checkpoint), Path(repo)
    if not checkpoint.is_file():
        raise PenetError(f"PENet checkpoint not found: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != PENET_WEIGHT_SHA256:
        raise PenetError(f"PENet checkpoint SHA-256 mismatch: {checkpoint}")
    if not (repo / "models" / "penet_classifier.py").is_file():
        raise PenetError(f"PENet repository not found: {repo}")
    # The released archive pickles objects from the repository's own `util` package, which
    # imports cv2 at module scope. cv2 is a hard requirement of the preprocessing anyway,
    # so a missing one is reported here rather than as an opaque unpickling failure.
    _require_cv2()
    sys.path.insert(0, str(repo.resolve()))
    try:
        from models.penet_classifier import PENetClassifier

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = dict(payload.get("model_args") or {})
        if not args:
            raise PenetError(f"{checkpoint} carries no model_args")
        model = PENetClassifier(**args)
        state = {key.removeprefix("module."): value for key, value in payload["model_state"].items()}
        model.load_state_dict(state, strict=True)
    finally:
        if sys.path and sys.path[0] == str(repo.resolve()):
            sys.path.pop(0)
    model.eval()
    return model, {
        "model_name": str(payload.get("model_name") or ""),
        "model_args": args,
        "checkpoint_info": {k: float(v) if hasattr(v, "__float__") else v
                            for k, v in dict(payload.get("ckpt_info") or {}).items()},
        "tensors": len(state),
        "weight_sha256": PENET_WEIGHT_SHA256,
    }


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise PenetError(
            "PENet preprocessing needs opencv (cv2.INTER_AREA slice resize). Install "
            "opencv-python-headless; approximating that resize would change the score."
        ) from exc
    return cv2


def read_series_hu(path: str | Path, slice_order: str = "superior_to_inferior") -> np.ndarray:
    """Raw Hounsfield slices as (slices, height, width), in reading order."""
    import nibabel as nib

    source = Path(path)
    if not source.is_file():
        raise PenetError(f"CT series not found: {source}")
    image = nib.as_closest_canonical(nib.load(str(source)))
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise PenetError(f"expected a 3-D CT, got {data.shape}")
    # Canonical RAS arrays are (x: ->Right, y: ->Anterior, z: ->Superior). PENet was trained
    # on stacked DICOM pixel_arrays (scripts/create_hdf5.py -> util.dcm_to_raw, no reorient),
    # whose rows run anterior->posterior and columns patient right->left (LPS). Its training
    # used horizontal flips but never vertical ones (do_vflip=False), so an upside-down slice
    # is out of distribution: reverse both in-plane axes to reproduce the DICOM layout.
    volume = np.transpose(data, (2, 1, 0))[:, ::-1, ::-1]
    if slice_order == "superior_to_inferior":
        volume = volume[::-1]
    elif slice_order != "inferior_to_superior":
        raise PenetError(f"unknown slice_order {slice_order!r}")
    return np.ascontiguousarray(volume)


def _resize_and_crop(window: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    resized = np.stack(
        [cv2.resize(s, tuple(RESIZE_SHAPE), interpolation=cv2.INTER_AREA) for s in window]
    )
    row_margin = max(0, resized.shape[-2] - CROP_SHAPE[-2])
    col_margin = max(0, resized.shape[-1] - CROP_SHAPE[-1])
    row, col = row_margin // 2, col_margin // 2        # centre crop at test time
    return resized[:, row : row + CROP_SHAPE[-2], col : col + CROP_SHAPE[-1]]


def _normalize(window: np.ndarray) -> np.ndarray:
    scaled = (window - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN)
    return np.clip(scaled, 0.0, 1.0) - CONTRAST_HU_MEAN


def series_windows(volume: np.ndarray) -> Iterator[np.ndarray]:
    """Non-overlapping 32-slice windows, each (1, 32, 208, 208), air-padded at the end."""
    if volume.ndim != 3 or volume.shape[0] < 1:
        raise PenetError(f"expected (slices, h, w) Hounsfield volume, got {volume.shape}")
    total = volume.shape[0]
    count = total // NUM_SLICES + (1 if total % NUM_SLICES else 0)
    for index in range(count):
        start = index * NUM_SLICES
        window = volume[start : start + NUM_SLICES]
        missing = NUM_SLICES - window.shape[0]
        if missing > 0:
            window = np.pad(window, ((0, missing), (0, 0), (0, 0)), constant_values=AIR_HU_VAL)
        yield _normalize(_resize_and_crop(window))[None].astype(np.float32)
