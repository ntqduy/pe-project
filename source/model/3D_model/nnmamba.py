"""nnMamba4cls (Gong et al. 2024) from third_party/repos/nnMamba, unmodified.

Upstream publishes no classification weights, so this arm always trains from scratch (the
log says so). The embedding reproduces upstream exactly: the spatial means of stages c2, c3
and c4 concatenated (2C + 4C + 8C = 14C = 448 for C = 32), before its own MLP, which the
shared projection + head replace. The Grad-CAM target is c4, the deepest map, so
``feature_dim`` (embedding, 14C) and ``feature_map_dim`` (c4 channels, 8C = 256) differ; ROI /
anatomy pooling of the map uses the latter.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from source.model.base import BaselineEncoder, format_pretrained_report, intensity_from_config, scratch_report

from ._thirdparty import import_from_repo, require_mamba_ssm


class NNMambaCore(nn.Module):
    def __init__(self, channels: int = 32, blocks: int = 3):
        super().__init__()
        require_mamba_ssm()
        module = import_from_repo("nnMamba", "nnMamba4cls")
        self.net = module.nnMambaEncoder(in_ch=1, channels=int(channels), blocks=int(blocks), number_classes=1)
        self.net.mlp = nn.Identity()  # replaced by the shared projection + head
        self.channels = int(channels)

    def forward(self, volume: Tensor) -> dict[str, Any]:
        net = self.net
        c1 = net.in_conv(volume)
        c1_s = net.mamba_layer_stem(c1) + c1
        c2 = net.layer1(c1_s)
        c3 = net.layer2(c2)
        c4 = net.layer3(c3)
        embedding = torch.cat([level.float().mean(dim=(2, 3, 4)) for level in (c2, c3, c4)], dim=1)
        return {"feature_map": c4, "global_embedding": embedding, "pyramid": [c2, c3, c4],
                "pooling": "concat_mean_c2_c3_c4"}


def build_nnmamba_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    channels = int(config.get("channels", 32))
    core = NNMambaCore(channels=channels, blocks=int(config.get("blocks", 3)))
    encoder = BaselineEncoder(
        core,
        14 * channels,
        "nnMamba4cls 3D",
        intensity=intensity_from_config(config.get("intensity"), default_mode="unit"),
        architecture_note="Model 3D: nnMamba4cls (conv + Mamba stem), embedding = concat mean(c2,c3,c4); Grad-CAM trên c4",
        lora_target_modules=tuple(config.get("lora_target_modules") or ("layer3",)),
        feature_map_dim=8 * channels,
    )
    encoder.pretrained_report = scratch_report("upstream nnMamba publishes no classification weights")
    print(format_pretrained_report(encoder.backbone_name, encoder.pretrained_report), flush=True)
    return encoder
