"""Shared contract of the baseline encoders: intensity mapping, wrapper, weight-load report.

Input convention (fixed by ``source/data/profiles/_common.yaml``): every cached volume is
``[B, 1, x, y, z]`` in RAS order, 1.5 mm isotropic, HU clipped to [-1000, 1000] and min-max
scaled to [0, 1]. An encoder never sees anything else, so each one converts that tensor to
the intensity convention its weights were pre-trained with (``IntensityAdapter``) and keeps
its feature map in the input's axis order, so Grad-CAM overlays line up with the cache.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from source.components.encoders.image.base import BaseImageEncoder, ImageFeatures

CACHE_HU_RANGE = (-1000.0, 1000.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# PENet's live contrast window and mean (see source/components/encoders/image/penet_zeroshot.py).
PENET_HU = (-100.0, 900.0)
PENET_MEAN = 0.15897


class IntensityAdapter(nn.Module):
    """Map the cache's [0, 1] (of ``input_hu_range``) to one pre-training convention.

    modes
      unit      unchanged: [0, 1] of [-1000, 1000] HU (Swin-UNETR SSL, scratch models)
      window    clip HU to ``window`` and rescale to [0, 1]; then ``(v - mean) / std`` when
                mean/std are given (ImageNet backbones use the ImageNet statistics)
      zscore    per-volume z-score of the HU-clipped window (MedicalNet, Mamba-MAE)
      penet     PENet: clip to [-100, 900], rescale to [0, 1], subtract 0.15897

    No parameters or buffers, so checkpoints of every mode stay loadable across modes.
    """

    def __init__(
        self,
        mode: str = "unit",
        *,
        input_hu_range: Sequence[float] = CACHE_HU_RANGE,
        window: Sequence[float] | None = None,
        mean: float | Sequence[float] | None = None,
        std: float | Sequence[float] | None = None,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"unit", "window", "zscore", "penet"}:
            raise ValueError(f"unknown intensity mode {mode!r}")
        self.input_hu_range = tuple(float(value) for value in input_hu_range)
        self.window = tuple(float(value) for value in window) if window else None
        if self.mode == "window" and self.window is None:
            raise ValueError("intensity mode 'window' needs window: [low_hu, high_hu]")
        self.mean = mean
        self.std = std

    def _hu(self, volume: Tensor) -> Tensor:
        low, high = self.input_hu_range
        return volume.float() * (high - low) + low

    @staticmethod
    def _normalize(values: Tensor, mean: Any, std: Any) -> Tensor:
        if mean is None or std is None:
            return values
        mean_t = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
        std_t = torch.as_tensor(std, dtype=values.dtype, device=values.device)
        if mean_t.ndim == 1 and mean_t.numel() == values.shape[1] and values.ndim >= 3:
            shape = (1, -1) + (1,) * (values.ndim - 2)
            mean_t, std_t = mean_t.view(shape), std_t.view(shape)
        elif mean_t.ndim == 1:
            mean_t, std_t = mean_t.mean(), std_t.mean()
        return (values - mean_t) / std_t

    def forward(self, volume: Tensor) -> Tensor:
        if self.mode == "unit":
            return volume.float()
        hu = self._hu(volume)
        if self.mode == "window":
            low, high = self.window  # type: ignore[misc]
            scaled = (hu.clamp(low, high) - low) / (high - low)
            return self._normalize(scaled, self.mean, self.std)
        if self.mode == "penet":
            low, high = PENET_HU
            return ((hu - low) / (high - low)).clamp(0.0, 1.0) - PENET_MEAN
        # zscore: per sample over every spatial voxel of the (optionally windowed) HU volume.
        if self.window is not None:
            hu = hu.clamp(*self.window)
        dims = tuple(range(1, hu.ndim))
        mean = hu.mean(dim=dims, keepdim=True)
        std = hu.std(dim=dims, keepdim=True).clamp_min(1e-6)
        return (hu - mean) / std


def intensity_from_config(config: Mapping[str, Any] | None, default_mode: str = "unit") -> IntensityAdapter:
    options = dict(config or {})
    return IntensityAdapter(
        str(options.get("mode", default_mode)),
        input_hu_range=options.get("input_hu_range") or CACHE_HU_RANGE,
        window=options.get("window"),
        mean=options.get("mean"),
        std=options.get("std"),
    )


class BaselineEncoder(BaseImageEncoder):
    """``BaseImageEncoder`` around a core module returning ``{feature_map, global_embedding}``.

    The core is kept as ``self.model`` and called through ``__call__``: the Grad-CAM preview
    (``source/engine/task_artifacts.py``) hooks exactly ``encoder.model`` and reads the 5-D
    ``feature_map`` from its output, so every baseline gets the same CAM machinery as CT-FM.
    ``feature_map`` is always in the input's (RAS) axis order.
    """

    def __init__(
        self,
        core: nn.Module,
        feature_dim: int,
        backbone_name: str,
        *,
        intensity: IntensityAdapter | None = None,
        architecture_note: str = "",
        lora_target_modules: Sequence[str] = (),
        feature_map_dim: int | None = None,
    ):
        super().__init__()
        self.model = core
        self.preprocess = intensity or IntensityAdapter("unit")
        self.feature_dim = int(feature_dim)
        if feature_map_dim is not None:
            # Only when global_embedding is not the spatial mean of feature_map (nnMamba).
            self.feature_map_dim = int(feature_map_dim)
        self.backbone_name = str(backbone_name)
        self.architecture_note = architecture_note
        self.lora_target_modules = tuple(lora_target_modules)
        self.pretrained_report: dict[str, Any] = {"status": "not_requested"}

    def forward_features(self, volume: Tensor) -> ImageFeatures:
        output = self.model(self.preprocess(volume))
        if not isinstance(output, Mapping) or "feature_map" not in output:
            raise RuntimeError(f"{self.backbone_name} core must return a mapping with feature_map")
        feature_map = output["feature_map"]
        embedding = output.get("global_embedding")
        if embedding is None:
            embedding = feature_map.float().mean(dim=(2, 3, 4)).to(feature_map.dtype)
        if feature_map.ndim != 5 or embedding.ndim != 2:
            raise RuntimeError(
                f"{self.backbone_name}: feature_map must be 5-D and global_embedding 2-D, got "
                f"{tuple(feature_map.shape)} / {tuple(embedding.shape)}"
            )
        metadata = {"backbone": self.backbone_name, "pooling": output.get("pooling", "mean")}
        metadata.update(dict(output.get("metadata") or {}))
        return ImageFeatures(
            feature_map=feature_map,
            global_embedding=embedding,
            pyramid=tuple(output.get("pyramid") or ()),
            metadata=metadata,
        )

    def load_pretrained_weights(self, checkpoint: str, strict: bool = True) -> dict[str, Any]:
        from pathlib import Path

        from source.components.encoders.image.external import load_state_file

        payload = load_state_file(Path(checkpoint))
        state = unwrap_state_dict(payload)
        return load_state_with_report(self.model, state, source=str(checkpoint), strict=strict)


# --------------------------------------------------------------------------------------
# Weight loading with an explicit, loggable report
# --------------------------------------------------------------------------------------


def unwrap_state_dict(payload: Any) -> dict[str, Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state", "state_dict", "model", "net", "module"):
            if key in payload and isinstance(payload[key], Mapping):
                payload = payload[key]
                break
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint does not contain a state dict")
    return {str(key).removeprefix("module."): value for key, value in payload.items() if isinstance(value, Tensor)}


def load_state_with_report(
    module: nn.Module,
    state: Mapping[str, Tensor],
    *,
    source: str,
    strict: bool = False,
    notes: Sequence[str] = (),
    minimum_fraction: float = 0.5,
) -> dict[str, Any]:
    """Copy every tensor whose name and shape match; report the rest instead of hiding it.

    ``strict`` raises on any missing/unexpected/shape-mismatched key. Otherwise the load
    still fails loudly when fewer than ``minimum_fraction`` of the module's tensors could be
    matched: that is a wrong checkpoint, not a partial adaptation.
    """
    own = module.state_dict()
    matched: dict[str, Tensor] = {}
    mismatched: list[str] = []
    for key, value in state.items():
        if key in own:
            if tuple(own[key].shape) == tuple(value.shape):
                matched[key] = value.to(own[key].dtype)
            else:
                mismatched.append(f"{key}: ckpt {tuple(value.shape)} vs model {tuple(own[key].shape)}")
    missing = sorted(set(own) - set(matched))
    unexpected = sorted(set(state) - set(own))
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            f"strict load of {source} failed: missing={missing[:8]} unexpected={unexpected[:8]} "
            f"shape_mismatch={mismatched[:8]}"
        )
    fraction = len(matched) / max(len(own), 1)
    if fraction < minimum_fraction:
        raise RuntimeError(
            f"pretrained weights {source} match only {len(matched)}/{len(own)} tensors "
            f"({100 * fraction:.1f}%); refusing a mostly-random 'pretrained' model. "
            f"first missing={missing[:6]} first mismatched={mismatched[:4]}"
        )
    module.load_state_dict(matched, strict=False)
    return {
        "status": "loaded",
        "source": source,
        "matched_tensors": len(matched),
        "model_tensors": len(own),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": mismatched,
        "notes": list(notes),
    }


def scratch_report(reason: str) -> dict[str, Any]:
    return {"status": "scratch", "source": None, "reason": reason}


def format_pretrained_report(name: str, report: Mapping[str, Any]) -> str:
    """One human-readable banner line for logs and the weight-status table."""
    status = str(report.get("status"))
    if status == "loaded":
        text = (
            f"PRETRAINED LOADED | {name} | source={report.get('source')} | "
            f"matched {report.get('matched_tensors')}/{report.get('model_tensors')} tensors"
        )
        missing = report.get("missing_keys") or []
        if missing:
            text += f" | not in checkpoint (random init): {len(missing)}"
        mismatch = report.get("shape_mismatch") or []
        if mismatch:
            text += f" | shape-mismatched (skipped): {len(mismatch)}"
        if report.get("notes"):
            text += " | adapted: " + "; ".join(str(note) for note in report["notes"])
        return text
    if status == "scratch":
        return f"NO PRETRAINED - training from scratch | {name} | reason: {report.get('reason')}"
    if status == "delegated":
        return f"PRETRAINED ({report.get('source')}) | {name} | {report.get('reason')}"
    return f"PRETRAINED STATUS {status} | {name} | {json.dumps(dict(report), default=str)[:300]}"


def pretrained_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """``model.pretrained`` merged with the global ``model.load_pretrained`` switch."""
    options = dict(config.get("pretrained") or {})
    options.setdefault("enabled", True)
    if config.get("load_pretrained") is False:
        options["enabled"] = False
    options.setdefault("required", False)
    return options


def resolve_pretrained(
    encoder: BaselineEncoder,
    config: Mapping[str, Any],
    loader: Any,
) -> BaselineEncoder:
    """Run ``loader(options) -> report`` honouring ``enabled`` / ``required``.

    ``required: false`` turns a failed load (no network, missing file) into an explicit
    "NO PRETRAINED - training from scratch" report; ``required: true`` re-raises it.
    """
    options = pretrained_options(config)
    if not options["enabled"]:
        encoder.pretrained_report = scratch_report("disabled by config (model.load_pretrained/pretrained.enabled)")
        return encoder
    try:
        encoder.pretrained_report = dict(loader(options))
    except Exception as exc:  # noqa: BLE001 - reported, and re-raised when required
        if options["required"]:
            raise
        encoder.pretrained_report = scratch_report(f"load failed: {type(exc).__name__}: {exc}")
    print(format_pretrained_report(encoder.backbone_name, encoder.pretrained_report), flush=True)
    return encoder


__all__ = [
    "BaselineEncoder",
    "CACHE_HU_RANGE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IntensityAdapter",
    "format_pretrained_report",
    "intensity_from_config",
    "load_state_with_report",
    "pretrained_options",
    "resolve_pretrained",
    "scratch_report",
    "unwrap_state_dict",
]
