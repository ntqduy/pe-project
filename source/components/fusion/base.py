from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import Tensor, nn


class FusionModule(nn.Module, ABC):
    """Common mapping-based interface for independently publishable fusion modules."""

    feature_names: tuple[str, ...]
    output_dim: int

    @abstractmethod
    def forward(
        self,
        features: Mapping[str, Tensor],
        availability: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError


def ordered_features(
    feature_names: tuple[str, ...],
    features: Mapping[str, Tensor],
) -> list[Tensor]:
    missing = [name for name in feature_names if name not in features]
    extra = [name for name in features if name not in feature_names]
    if missing or extra:
        raise KeyError(f"fusion feature mismatch: missing={missing} extra={extra}")
    values = [features[name] for name in feature_names]
    if not values:
        raise ValueError("fusion requires at least one feature")
    batch = values[0].shape[0]
    if any(value.ndim != 2 or value.shape[0] != batch for value in values):
        raise ValueError("fusion inputs must be [B,D] tensors with a common batch size")
    return values
