from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .base import ImageFeatures
from .external import InspectedExternalEncoder, ThirdPartyIntegrationError, build_inspected_external


def build_ct_fm(config: Mapping[str, Any]) -> InspectedExternalEncoder:
    return build_inspected_external(config, "CT-FM")


def build_ct_fm_backbone(
    spatial_dims: int = 3,
    init_filters: int = 32,
    in_channels: int = 1,
    blocks_down: Sequence[int] = (1, 2, 2, 4, 4),
) -> nn.Module:
    """SegResEncoder matching project-lighter/ct_fm_feature_extractor.

    Defaults are the values read from that checkpoint's own config.json, verified by a strict
    load of all 161 tensors. `head_module` is deliberately left unset: upstream's
    scripts/feature_extractor.py pools inside the model, which would discard the feature map
    the project contract requires. The adapter below pools instead.
    """
    from monai.networks.nets.segresnet_ds import SegResEncoder

    return SegResEncoder(
        spatial_dims=int(spatial_dims),
        init_filters=int(init_filters),
        in_channels=int(in_channels),
        blocks_down=tuple(int(value) for value in blocks_down),
    )


def ct_fm_output_adapter(outputs: Any) -> ImageFeatures:
    if isinstance(outputs, Tensor):
        pyramid: tuple[Tensor, ...] = (outputs,)
    elif isinstance(outputs, (list, tuple)) and outputs:
        pyramid = tuple(outputs)
    else:
        raise ThirdPartyIntegrationError("CT-FM encoder must return a tensor or a non-empty feature pyramid")
    feature_map = pyramid[-1]
    global_embedding = torch.nn.functional.adaptive_avg_pool3d(feature_map, 1).flatten(start_dim=1)
    return ImageFeatures(
        feature_map=feature_map,
        global_embedding=global_embedding,
        pyramid=pyramid,
        metadata={"backbone": "ct_fm", "pooling": "adaptive_avg_pool3d"},
    )
