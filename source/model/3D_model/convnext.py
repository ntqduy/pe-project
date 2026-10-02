"""ConvNeXt 3D, parameter-for-parameter the timm ConvNeXt with 3-D kernels.

Module and parameter names mirror ``timm.models.convnext`` (stem.0/1, stages.i.downsample,
stages.i.blocks.j.{conv_dw, norm, mlp.fc1, mlp.fc2, gamma}), so ImageNet ConvNeXt weights
inflate onto it key by key: 4x4 stem -> 4x4x4, 2x2 downsampling -> 2x2x2, 7x7 depthwise ->
7x7x7 (2-D kernel in the axial x-y plane, repeated along z and divided by its z size; see
source/model/inflate.py), norms and MLPs copied unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from source.model.base import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BaselineEncoder,
    intensity_from_config,
    load_state_with_report,
    resolve_pretrained,
)
from source.model.inflate import inflate_state_dict, timm_state_dict

VARIANTS = {
    "tiny": ((3, 3, 9, 3), (96, 192, 384, 768), "convnext_tiny.fb_in22k_ft_in1k"),
    "small": ((3, 3, 27, 3), (96, 192, 384, 768), "convnext_small.fb_in22k_ft_in1k"),
}


class LayerNorm3d(nn.LayerNorm):
    """Channel-first LayerNorm (timm LayerNorm2d with one more spatial axis)."""

    def forward(self, values: Tensor) -> Tensor:
        values = values.permute(0, 2, 3, 4, 1)
        values = nn.functional.layer_norm(values, self.normalized_shape, self.weight, self.bias, self.eps)
        return values.permute(0, 4, 1, 2, 3)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, values: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(values)))


class ConvNeXtBlock3d(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale: float = 1e-6):
        super().__init__()
        self.conv_dw = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, 4 * dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = float(drop_path)

    def forward(self, values: Tensor) -> Tensor:
        shortcut = values
        values = self.conv_dw(values).permute(0, 2, 3, 4, 1)
        values = self.mlp(self.norm(values)) * self.gamma
        values = values.permute(0, 4, 1, 2, 3)
        if self.training and self.drop_path > 0:
            keep = torch.rand(values.shape[0], 1, 1, 1, 1, device=values.device) >= self.drop_path
            values = values * keep.to(values.dtype) / (1 - self.drop_path)
        return shortcut + values


class ConvNeXtStage3d(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, depth: int, *, downsample: bool, drop_paths: Sequence[float]):
        super().__init__()
        self.downsample = (
            nn.Sequential(LayerNorm3d(in_dim, eps=1e-6), nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2))
            if downsample
            else nn.Identity()
        )
        self.blocks = nn.Sequential(*(ConvNeXtBlock3d(out_dim, drop_paths[index]) for index in range(depth)))


class ConvNeXt3dCore(nn.Module):
    def __init__(self, depths: Sequence[int], dims: Sequence[int], drop_path_rate: float = 0.1, gradient_checkpointing: bool = True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, dims[0], kernel_size=4, stride=4), LayerNorm3d(dims[0], eps=1e-6))
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        stages, cursor, previous = [], 0, dims[0]
        for index, (depth, dim) in enumerate(zip(depths, dims)):
            stages.append(
                ConvNeXtStage3d(previous, dim, depth, downsample=index > 0, drop_paths=rates[cursor : cursor + depth])
            )
            cursor += depth
            previous = dim
        self.stages = nn.Sequential(*stages)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, volume: Tensor) -> dict[str, Tensor]:
        values = self.stem(volume)
        pyramid = []
        for stage in self.stages:
            values = stage.downsample(values)
            for block in stage.blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    values = checkpoint(block, values, use_reentrant=False)
                else:
                    values = block(values)
            pyramid.append(values)
        return {"feature_map": values, "pyramid": pyramid}


def build_convnext_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    variant = str(config.get("variant") or "tiny")
    depths, dims, default_source = VARIANTS[variant]
    core = ConvNeXt3dCore(
        depths,
        dims,
        drop_path_rate=float(config.get("drop_path_rate", 0.1)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
    )
    default_intensity = {"mode": "window", "window": [-250.0, 450.0], "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
    encoder = BaselineEncoder(
        core,
        dims[-1],
        f"ConvNeXt-{variant[0].upper()} 3D",
        intensity=intensity_from_config(config.get("intensity") or default_intensity),
        architecture_note=f"Model 3D: ConvNeXt-{variant} 3D (kernel 7x7x7), một forward trên cả volume; feature map = stage 4",
        # Stage 4's downsample conv and block MLP Linears; inject_lora skips the depthwise conv_dw.
        lora_target_modules=tuple(config.get("lora_target_modules") or ("stages.3",)),
    )

    def load(options: Mapping[str, Any]) -> dict[str, Any]:
        name = str(options.get("timm_name") or default_source)
        source_state, source = timm_state_dict(name)
        state, summary = inflate_state_dict(
            source_state, core.state_dict(), rename=lambda key: None if key.startswith(("head", "norm_pre")) else key
        )
        notes = [f"ImageNet 2D -> 3D inflation (copied {summary['copied']}, inflated {summary['inflated']})",
                 "RGB stem summed to 1 channel"]
        return load_state_with_report(core, state, source=source, notes=notes)

    return resolve_pretrained(encoder, config, load)
