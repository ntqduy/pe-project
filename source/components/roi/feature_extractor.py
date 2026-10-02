from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor, nn

from .pooling import mask_guided_pool


class ROIFeatureExtractor(nn.Module):
    def __init__(self, regions: tuple[str, ...] = ("heart", "pa", "lung")):
        super().__init__()
        self.regions = regions

    def forward(
        self,
        feature_map: Tensor,
        masks: Mapping[str, Tensor],
        *,
        slice_groups: Sequence[Sequence[int]] | None = None,
        slice_depth: int | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """``slice_groups``/``slice_depth``: the axial slices behind each instance of a
        slice-MIL map [B, C, h, w, N] (see ``mask_guided_pool``); None for a 3-D map."""
        features: dict[str, Tensor] = {}
        present: dict[str, Tensor] = {}
        for region in self.regions:
            if region not in masks:
                raise KeyError(f"required ROI mask is missing: {region}")
            features[region], present[region] = mask_guided_pool(
                feature_map, masks[region], slice_groups=slice_groups, slice_depth=slice_depth
            )
        return features, present
