"""Shared builder of every 2-D / 2.5-D slice-MIL baseline."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from source.model.base import IMAGENET_MEAN, IMAGENET_STD, BaselineEncoder, format_pretrained_report, intensity_from_config, pretrained_options, scratch_report

from .mil import SliceMILCore
from .timm2d import Timm2DBackbone, pretrained_summary

DEFAULT_INTENSITY = {"mode": "window", "window": [-250.0, 450.0], "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}


def build_slice_mil_encoder(config: Mapping[str, Any], *, default_timm_name: str, display_name: str,
                            default_lora_targets: tuple[str, ...] = ()) -> BaselineEncoder:
    timm_name = str(config.get("timm_name") or default_timm_name)
    slice_mode = str(config.get("slice_mode") or "2d").lower()
    in_chans = 3 if slice_mode in {"2.5d", "25d"} else 1
    mil = dict(config.get("mil") or {})
    slice_size = int(mil.get("slice_size", 224))
    options = pretrained_options(config)
    if options["enabled"]:
        try:
            backbone = Timm2DBackbone(timm_name, in_chans=in_chans, img_size=slice_size, pretrained=True)
            report = dict(pretrained_summary(backbone))
        except Exception as exc:  # noqa: BLE001
            if options["required"]:
                raise
            backbone = Timm2DBackbone(timm_name, in_chans=in_chans, img_size=slice_size, pretrained=False)
            report = scratch_report(f"timm download failed: {type(exc).__name__}: {exc}")
    else:
        backbone = Timm2DBackbone(timm_name, in_chans=in_chans, img_size=slice_size, pretrained=False)
        report = scratch_report("disabled by config (model.load_pretrained/pretrained.enabled)")
    intensity = dict(config.get("intensity") or DEFAULT_INTENSITY)
    normalize_mean, normalize_std = intensity.pop("mean", None), intensity.pop("std", None)
    core = SliceMILCore(backbone, backbone.num_features, mode=slice_mode,
                        num_slices=int(mil.get("num_slices", 32)), slice_size=slice_size,
                        slice_offset=int(mil.get("slice_offset", 1)), pooling=str(mil.get("pooling", "attention")),
                        attention_hidden=int(mil.get("attention_hidden", 128)), chunk_size=int(mil.get("chunk_size", 32)),
                        gradient_checkpointing=bool(mil.get("gradient_checkpointing", True)),
                        normalize_mean=normalize_mean, normalize_std=normalize_std)
    label = "2.5D" if in_chans == 3 else "2D"
    encoder = BaselineEncoder(
        core, backbone.num_features, f"{display_name} {label} slice-MIL", intensity=intensity_from_config(intensity),
        architecture_note=(f"Model {label} slice-MIL: {core.num_slices} uniformly spaced axial slices"
                           + ("; each instance has three adjacent slices as channels" if in_chans == 3 else "")
                           + f"; 2-D backbone {timm_name} on {slice_size}x{slice_size}, pooled with {core.pooling} attention"),
        lora_target_modules=tuple(config.get("lora_target_modules") or default_lora_targets),
    )
    # The gated-attention MIL pool is new (no ImageNet weights): it trains under LoRA too.
    encoder.peft_trainable_modules = ("model.attention",) if core.attention is not None else ()
    encoder.pretrained_report = report
    print(format_pretrained_report(encoder.backbone_name, report), flush=True)
    return encoder


__all__ = ["DEFAULT_INTENSITY", "build_slice_mil_encoder"]
