from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from source.components.adapters.organ import OrganAdapterBank
from source.components.encoders.image.base import BaseImageEncoder
from source.components.roi.feature_extractor import ROIFeatureExtractor


@dataclass(frozen=True)
class AnatomyFeatureOutput:
    """Region-pooled representations from a shared full-CTPA feature map.

    Regional values are not strict organ-only information: every ``z_*`` is pooled from
    a spatial feature map produced by encoding the complete input CTPA.
    """

    z_global: Tensor
    z_heart: Tensor | None
    z_pa: Tensor | None
    z_lung: Tensor | None
    adapted_global: Tensor | None
    adapted_heart: Tensor | None
    adapted_pa: Tensor | None
    adapted_lung: Tensor | None
    available: dict[str, Tensor]

    def as_dict(self) -> dict[str, Tensor | None]:
        return {
            "z_global": self.z_global,
            "z_heart": self.z_heart,
            "z_pa": self.z_pa,
            "z_lung": self.z_lung,
            "adapted_global": self.adapted_global,
            "adapted_heart": self.adapted_heart,
            "adapted_pa": self.adapted_pa,
            "adapted_lung": self.adapted_lung,
        }

    def fusion_inputs(self, feature_names: tuple[str, ...]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        values: dict[str, Tensor] = {}
        present: dict[str, Tensor] = {}
        for name in feature_names:
            value = getattr(self, f"adapted_{name}")
            if value is None:
                raise KeyError(f"adapted anatomy feature is unavailable: {name}")
            values[name] = value
            present[name] = self.available[name]
        return values, present


def global_average_pool(feature_map: Tensor) -> Tensor:
    if feature_map.ndim != 5:
        raise ValueError(f"feature map must be [B,C,D,H,W], got {tuple(feature_map.shape)}")
    return feature_map.mean(dim=(2, 3, 4))


def extract_anatomy_features(
    image_encoder: BaseImageEncoder,
    roi: ROIFeatureExtractor,
    organ_adapters: OrganAdapterBank,
    volume: Tensor,
    masks: dict[str, Tensor] | Any,
) -> AnatomyFeatureOutput:
    """Encode a full CTPA once, then pool and adapt global/organ representations."""

    image = image_encoder.forward_features(volume)
    z_global = global_average_pool(image.feature_map)
    if z_global.shape[1] != int(image_encoder.feature_dim):
        raise ValueError(
            "encoder feature_dim must equal spatial feature-map channels: "
            f"configured={image_encoder.feature_dim} observed={z_global.shape[1]}"
        )
    regional, regional_present = roi(image.feature_map, masks)
    raw = {"global": z_global, **regional}
    available = {
        "global": torch.ones(volume.shape[0], dtype=torch.bool, device=volume.device),
        **regional_present,
    }
    adapted, adapted_present = organ_adapters(raw, available)
    available.update(adapted_present)

    return AnatomyFeatureOutput(
        z_global=z_global,
        z_heart=regional.get("heart"),
        z_pa=regional.get("pa"),
        z_lung=regional.get("lung"),
        adapted_global=adapted.get("global"),
        adapted_heart=adapted.get("heart"),
        adapted_pa=adapted.get("pa"),
        adapted_lung=adapted.get("lung"),
        available=available,
    )

