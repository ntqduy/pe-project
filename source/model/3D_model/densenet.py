"""DenseNet-121 3D (MONAI) initialised by inflating ImageNet DenseNet-121.

There is no public volumetric DenseNet-121 checkpoint, so the torchvision ImageNet weights
(timm ``densenet121.tv_in1k``, torchvision key names) are inflated I3D-style. MONAI names
the inside of each dense layer ``layers.<name>``; torchvision does not, hence the rename.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from source.model.base import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BaselineEncoder,
    intensity_from_config,
    load_state_with_report,
    resolve_pretrained,
)
from source.model.inflate import inflate_state_dict, timm_state_dict

DEFAULT_SOURCE = "densenet121.tv_in1k"
_DENSE_LAYER = re.compile(r"^(features\.denseblock\d+\.denselayer\d+)\.(norm1|norm2|conv1|conv2)\.(.+)$")


class MonaiDenseNetCore(nn.Module):
    def __init__(self):
        super().__init__()
        from monai.networks.nets import DenseNet121

        self.net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
        # Replaced by the shared projection + head; an unused Linear would stall DDP.
        self.net.class_layers = nn.Identity()

    def forward(self, volume: Tensor) -> dict[str, Tensor]:
        # MONAI's features end with norm5; class_layers would add ReLU + pooling.
        return {"feature_map": nn.functional.relu(self.net.features(volume))}


def _torchvision_to_monai(key: str) -> str | None:
    if key.startswith("classifier"):
        return None
    match = _DENSE_LAYER.match(key)
    if match:
        return f"{match.group(1)}.layers.{match.group(2)}.{match.group(3)}"
    return key


def build_densenet121_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    core = MonaiDenseNetCore()
    default_intensity = {"mode": "window", "window": [-250.0, 450.0], "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
    encoder = BaselineEncoder(
        core,
        1024,
        "DenseNet-121 3D",
        intensity=intensity_from_config(config.get("intensity") or default_intensity),
        architecture_note="Model 3D: MONAI DenseNet-121, một forward trên cả volume; feature map = features.norm5 (ReLU)",
        lora_target_modules=tuple(config.get("lora_target_modules") or ("denseblock4",)),
    )

    def load(options: Mapping[str, Any]) -> dict[str, Any]:
        name = str(options.get("timm_name") or DEFAULT_SOURCE)
        source_state, source = timm_state_dict(name)
        state, summary = inflate_state_dict(source_state, core.net.state_dict(), rename=_torchvision_to_monai)
        notes = [f"ImageNet 2D -> 3D inflation (copied {summary['copied']}, inflated {summary['inflated']})",
                 "RGB stem summed to 1 channel"]
        return load_state_with_report(core.net, state, source=source, notes=notes)

    return resolve_pretrained(encoder, config, load)
