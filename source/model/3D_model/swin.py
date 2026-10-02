"""Swin Transformer 3D: MONAI's Swin-UNETR encoder with its self-supervised CT weights.

``model_swinvit.pt`` (Tang et al., CVPR 2022) pre-trains exactly this SwinTransformer
(feature size 48, patch 2, window 7, depths 2/2/2/2) on ~5k CT volumes with intensities
scaled from [-1000, 1000] HU to [0, 1] - the same scaling as this project's cache, so the
default intensity mode is ``unit``. Only the ``swinViT`` encoder is kept; its deepest stage
(768 channels, 1/32 resolution) is the feature map.
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from source.model.base import BaselineEncoder, intensity_from_config, load_state_with_report, resolve_pretrained
from source.model.weights import SWIN_UNETR_SSL_URL, ensure_weight_file

DEFAULT_WEIGHT = "third_party/weights/baselines/swin_unetr_ssl/model_swinvit.pt"


class SwinTransformer3dCore(nn.Module):
    def __init__(self, feature_size: int = 48, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 drop_path_rate: float = 0.1, gradient_checkpointing: bool = True):
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer

        wanted = dict(
            in_chans=1, embed_dim=int(feature_size), window_size=(7, 7, 7), patch_size=(2, 2, 2),
            depths=tuple(depths), num_heads=tuple(num_heads), mlp_ratio=4.0, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=float(drop_path_rate),
            norm_layer=nn.LayerNorm, use_checkpoint=bool(gradient_checkpointing), spatial_dims=3,
        )
        accepted = set(inspect.signature(SwinTransformer.__init__).parameters)
        self.swinViT = SwinTransformer(**{key: value for key, value in wanted.items() if key in accepted})
        self.out_channels = int(feature_size) * 16

    def forward(self, volume: Tensor) -> dict[str, Tensor]:
        stages = self.swinViT(volume, True)   # normalize=True, as Swin-UNETR feeds its decoder
        return {"feature_map": stages[-1], "pyramid": list(stages)}


def build_swin_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    core = SwinTransformer3dCore(
        feature_size=int(config.get("feature_size", 48)),
        drop_path_rate=float(config.get("drop_path_rate", 0.1)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
    )
    encoder = BaselineEncoder(
        core,
        core.out_channels,
        "Swin 3D (Swin-UNETR SSL)",
        intensity=intensity_from_config(config.get("intensity"), default_mode="unit"),
        architecture_note="Model 3D: Swin Transformer 3D (window 7^3), một forward trên cả volume; feature map = stage 4",
        lora_target_modules=tuple(config.get("lora_target_modules") or ("attn.qkv", "attn.proj")),
    )

    def load(options: Mapping[str, Any]) -> dict[str, Any]:
        path = ensure_weight_file(options.get("path") or DEFAULT_WEIGHT, options.get("url") or SWIN_UNETR_SSL_URL,
                                  allow_download=bool(options.get("allow_download", True)))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
        # SSL heads (rotation / contrastive / reconstruction) are not part of the encoder.
        # The SSL code names the block MLP fc1/fc2; MONAI's MLPBlock calls them linear1/linear2
        # (MONAI's own SwinTransformerBlock.load_from applies the same mapping).
        state = {
            str(key).removeprefix("module.").replace(".mlp.fc1.", ".mlp.linear1.").replace(".mlp.fc2.", ".mlp.linear2."): value
            for key, value in state.items()
        }
        return load_state_with_report(core.swinViT, state, source=str(path),
                                      notes=["self-supervised heads dropped", "mlp.fc1/fc2 -> mlp.linear1/linear2"])

    return resolve_pretrained(encoder, config, load)
