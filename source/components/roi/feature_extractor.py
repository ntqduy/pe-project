from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from .pooling import mask_guided_pool


class ROIFeatureExtractor(nn.Module):
    def __init__(self, regions: tuple[str, ...] = ("heart", "pa", "lung")):
        super().__init__()
        self.regions = regions

    def forward(self, feature_map: Tensor, masks: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        features: dict[str, Tensor] = {}
        present: dict[str, Tensor] = {}
        for region in self.regions:
            if region not in masks:
                raise KeyError(f"required ROI mask is missing: {region}")
            features[region], present[region] = mask_guided_pool(feature_map, masks[region])
        return features, present
