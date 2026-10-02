"""ResNet-18 / ResNet-50 3D (MONAI) initialised from MedicalNet.

MedicalNet (Chen et al. 2019) pre-trained these exact MONAI ResNets on 23 segmentation
datasets (CT + MRI). MONAI downloads them from huggingface.co/TencentMedicalNet; their
contract fixes ``shortcut_type`` (A for 18/34, B for 50) and ``bias_downsample``, which is
why those two values are not configurable here. MedicalNet normalises each volume by its
own mean/std, hence the default ``zscore`` intensity.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from source.model.base import (
    BaselineEncoder,
    intensity_from_config,
    load_state_with_report,
    resolve_pretrained,
    unwrap_state_dict,
)

CHANNELS = {18: 512, 50: 2048}


class MonaiResNetCore(nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        from monai.networks.nets import resnet as monai_resnet
        from monai.networks.nets.resnet import get_medicalnet_pretrained_resnet_args

        bias_downsample, shortcut_type = get_medicalnet_pretrained_resnet_args(depth)
        factory = getattr(monai_resnet, f"resnet{depth}")
        self.net = factory(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            feed_forward=False,
            shortcut_type=shortcut_type,
            bias_downsample=bias_downsample,
            # MedicalNet's stem is a 7^3 conv with stride 2 (MONAI defaults to 1); the weights
            # load either way, but only stride 2 reproduces the receptive fields they learned.
            conv1_t_size=7,
            conv1_t_stride=2,
        )
        self.depth = depth

    def forward(self, volume: Tensor) -> dict[str, Tensor]:
        net = self.net
        x = net.act(net.bn1(net.conv1(volume)))
        if not net.no_max_pool:
            x = net.maxpool(x)
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return {"feature_map": x}


def _build(config: Mapping[str, Any], depth: int) -> BaselineEncoder:
    core = MonaiResNetCore(depth)
    encoder = BaselineEncoder(
        core,
        CHANNELS[depth],
        f"ResNet-{depth} 3D",
        intensity=intensity_from_config(config.get("intensity"), default_mode="zscore"),
        architecture_note=f"Model 3D: MONAI ResNet-{depth}, một forward trên cả volume; feature map = layer4",
        lora_target_modules=tuple(config.get("lora_target_modules") or ("layer3", "layer4")),
    )

    def load(options: Mapping[str, Any]) -> dict[str, Any]:
        if options.get("path"):
            import torch

            from source.model.weights import resolve_weight_path

            path = resolve_weight_path(options["path"])
            state = unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=False))
            source = str(path)
        else:
            from monai.networks.nets.resnet import get_pretrained_resnet_medicalnet

            state = unwrap_state_dict(get_pretrained_resnet_medicalnet(depth, device="cpu", datasets23=True))
            source = f"huggingface:TencentMedicalNet/MedicalNet-Resnet{depth} (23 datasets)"
        # MedicalNet checkpoints carry a segmentation decoder ("conv_seg"); only the encoder loads.
        state = {key: value for key, value in state.items() if not key.startswith("conv_seg")}
        return load_state_with_report(core.net, state, source=source, strict=False)

    return resolve_pretrained(encoder, config, load)


def build_resnet18_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    return _build(config, 18)


def build_resnet50_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    return _build(config, 50)
