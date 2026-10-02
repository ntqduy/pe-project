"""ViT 3D: timm ViT blocks with a 3-D patch embedding, ImageNet weights inflated.

The transformer blocks and final norm are timm's own modules with their ImageNet weights,
untouched. Only the two input-geometry pieces change:

* patch embedding  Conv2d(3, C, 16, 16) -> Conv3d(1, C, 16^3): RGB summed, kernel inflated
* positions        the 14 x 14 table is resized to the axial (x, y) token grid and repeated
                   along z (one table per 16-voxel axial slab); the class token is dropped

Both follow ``source/model/inflate.py``: the input is RAS [B, 1, x, y, z], so ImageNet's 2-D
structure lies in axial planes and is replicated along z.

The embedding is the mean of the final patch tokens, which makes the token grid
``[B, C, 8, 8, 8]`` (128^3 input) an exact Grad-CAM target layer.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from source.model.base import IMAGENET_MEAN, IMAGENET_STD, BaselineEncoder, intensity_from_config
from source.model.inflate import inflate_kernel, interpolate_positions_2d_to_3d

DEFAULT_TIMM_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"


class ViT3dCore(nn.Module):
    def __init__(self, timm_name: str, *, input_size: int, patch_size: int, pretrained: bool, gradient_checkpointing: bool):
        super().__init__()
        import timm

        self.load_report: dict | None = None
        if pretrained:
            from source.model.inflate import timm_create_with_report

            vit, self.load_report = timm_create_with_report(timm_name, num_classes=0)
        else:
            vit = timm.create_model(timm_name, pretrained=False, num_classes=0)
        dim = int(vit.embed_dim)
        self.grid = (input_size // patch_size,) * 3
        self.patch_embed = nn.Conv3d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1] * self.grid[2], dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = dim
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.inflation_notes: list[str] = []
        if pretrained:
            with torch.no_grad():
                weight2d = vit.patch_embed.proj.weight
                self.patch_embed.weight.copy_(inflate_kernel(weight2d, tuple(self.patch_embed.weight.shape)))
                self.patch_embed.bias.copy_(vit.patch_embed.proj.bias)
                prefix = int(getattr(vit, "num_prefix_tokens", 1))
                table = vit.pos_embed[:, prefix:]
                side = int(round(table.shape[1] ** 0.5))
                self.pos_embed.copy_(interpolate_positions_2d_to_3d(table, (side, side), self.grid))
            self.inflation_notes = [
                f"patch embedding Conv2d{tuple(weight2d.shape)} -> Conv3d{tuple(self.patch_embed.weight.shape)} (RGB summed, in the axial plane, averaged along z)",
                f"position table {side}x{side} -> {self.grid} (resized to the axial plane, repeated along z; cls token dropped)",
            ]

    def forward(self, volume: Tensor) -> dict[str, Tensor]:
        tokens = self.patch_embed(volume)                                  # [B, C, x, y, z]
        grid = tokens.shape[-3:]
        if tuple(grid) != self.grid:
            raise ValueError(f"ViT 3D was built for a {self.grid} token grid, got {tuple(grid)}; set model.input_size")
        tokens = tokens.flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        feature_map = tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, *grid)
        # The embedding is pooled from feature_map so the Grad-CAM target is on the gradient path.
        return {"feature_map": feature_map, "global_embedding": feature_map.float().mean(dim=(2, 3, 4)),
                "pooling": "patch_token_mean"}


def build_vit_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    from source.model.base import pretrained_options, scratch_report, format_pretrained_report

    timm_name = str(config.get("timm_name") or DEFAULT_TIMM_NAME)
    input_size = int(config.get("input_size", 128))
    patch_size = int(config.get("patch_size", 16))
    options = pretrained_options(config)
    kwargs = dict(input_size=input_size, patch_size=patch_size,
                  gradient_checkpointing=bool(config.get("gradient_checkpointing", True)))
    report: dict[str, Any]
    try:
        core = ViT3dCore(timm_name, pretrained=bool(options["enabled"]), **kwargs)
        if options["enabled"]:
            # Copied unchanged from the checkpoint: blocks.* and norm.*; adapted from it:
            # patch_embed (inflated) and pos_embed (resized) - listed in notes, not as matched.
            own = core.state_dict()
            timm_missing = set((core.load_report or {}).get("missing_keys") or [])
            copied = [key for key in own if key.startswith(("blocks.", "norm.")) and key not in timm_missing]
            adapted = [key for key in own if key.startswith(("patch_embed.", "pos_embed"))]
            report = {"status": "loaded", "source": (core.load_report or {}).get("source", f"timm:{timm_name}"),
                      "matched_tensors": len(copied), "model_tensors": len(own),
                      "missing_keys": sorted(set(own) - set(copied) - set(adapted)), "unexpected_keys": [],
                      "shape_mismatch": [],
                      "notes": [f"{len(adapted)} tensors adapted from the checkpoint: " + ", ".join(adapted)]
                      + core.inflation_notes}
        else:
            report = scratch_report("disabled by config (model.load_pretrained/pretrained.enabled)")
    except Exception as exc:  # noqa: BLE001
        if options["required"] or not options["enabled"]:
            raise
        core = ViT3dCore(timm_name, pretrained=False, **kwargs)
        report = scratch_report(f"timm download failed: {type(exc).__name__}: {exc}")
    default_intensity = {"mode": "window", "window": [-250.0, 450.0], "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
    encoder = BaselineEncoder(
        core,
        core.embed_dim,
        "ViT-S/16 3D",
        intensity=intensity_from_config(config.get("intensity") or default_intensity),
        architecture_note=(
            f"Model 3D: ViT, patch 3D {patch_size}^3 -> lưới token {core.grid}; Grad-CAM trên token map "
            "của block cuối (reshape về lưới 3D)"
        ),
        lora_target_modules=tuple(config.get("lora_target_modules") or ("attn.qkv", "attn.proj")),
    )
    encoder.pretrained_report = report
    print(format_pretrained_report(encoder.backbone_name, report), flush=True)
    return encoder
