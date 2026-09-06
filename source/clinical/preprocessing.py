from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class ClinicalPreprocessor(nn.Module):
    """Train-fit imputation and normalization stored as checkpoint buffers."""

    def __init__(
        self,
        input_dim: int,
        *,
        normalization: str = "zscore",
        imputation: str = "mean",
        constant_value: float = 0.0,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("clinical input_dim must be positive")
        normalization = normalization.lower()
        imputation = imputation.lower()
        if normalization not in {"none", "zscore", "standard"}:
            raise ValueError(f"unsupported clinical normalization: {normalization}")
        if imputation not in {"mean", "median", "constant"}:
            raise ValueError(f"unsupported clinical imputation: {imputation}")
        self.input_dim = int(input_dim)
        self.normalization = "zscore" if normalization == "standard" else normalization
        self.imputation = imputation
        self.constant_value = float(constant_value)
        self.register_buffer("center", torch.zeros(self.input_dim))
        self.register_buffer("scale", torch.ones(self.input_dim))
        self.register_buffer("fill", torch.zeros(self.input_dim))
        self.register_buffer("observed_count", torch.zeros(self.input_dim, dtype=torch.long))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, values: Tensor, *, split: str = "train") -> "ClinicalPreprocessor":
        if split != "train":
            raise ValueError("clinical preprocessing may only be fit on split='train'")
        values = torch.as_tensor(values, dtype=torch.float32, device=self.center.device)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"expected clinical matrix [N,{self.input_dim}], got {tuple(values.shape)}")
        finite = torch.isfinite(values)
        counts = finite.sum(dim=0)
        if bool((counts == 0).any()):
            columns = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"clinical columns contain no observed training values: {columns}")
        means = torch.where(finite, values, torch.zeros_like(values)).sum(dim=0) / counts
        if self.imputation == "mean":
            fill = means
        elif self.imputation == "median":
            fill = torch.stack([values[finite[:, index], index].median() for index in range(self.input_dim)])
        else:
            fill = torch.full_like(means, self.constant_value)
        centered = torch.where(finite, values - means, torch.zeros_like(values))
        variance = centered.square().sum(dim=0) / counts.clamp_min(1)
        scale = variance.sqrt().clamp_min(1e-6)
        self.center.copy_(means if self.normalization == "zscore" else torch.zeros_like(means))
        self.scale.copy_(scale if self.normalization == "zscore" else torch.ones_like(scale))
        self.fill.copy_(fill)
        self.observed_count.copy_(counts)
        self.fitted.fill_(True)
        return self

    def forward(self, values: Tensor, missing: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if not bool(self.fitted.item()):
            raise RuntimeError("clinical preprocessor has not been fit on training data")
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"expected clinical values [B,{self.input_dim}], got {tuple(values.shape)}")
        inferred = ~torch.isfinite(values)
        missing_mask = inferred if missing is None else missing.to(values.device).bool() | inferred
        if missing_mask.shape != values.shape:
            raise ValueError("clinical missingness mask must match values")
        fill = self.fill.to(values.device, values.dtype)
        complete = torch.where(missing_mask, fill.unsqueeze(0), values)
        normalized = (complete - self.center.to(values.device, values.dtype)) / self.scale.to(
            values.device, values.dtype
        )
        return normalized, missing_mask

    def export_state(self, columns: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
        names = list(columns or [str(index) for index in range(self.input_dim)])
        if len(names) != self.input_dim:
            raise ValueError("clinical preprocessing column list has the wrong length")
        return {
            "schema_version": 1,
            "fit_split": "train",
            "columns": names,
            "normalization": self.normalization,
            "imputation": self.imputation,
            "constant_value": self.constant_value,
            "fitted": bool(self.fitted.item()),
            "center": self.center.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
            "fill": self.fill.detach().cpu().tolist(),
            "observed_count": self.observed_count.detach().cpu().tolist(),
        }

    @torch.no_grad()
    def import_state(self, payload: Mapping[str, Any]) -> None:
        for name in ("center", "scale", "fill", "observed_count"):
            value = torch.as_tensor(payload[name], device=getattr(self, name).device, dtype=getattr(self, name).dtype)
            if value.shape != getattr(self, name).shape:
                raise ValueError(f"invalid clinical preprocessing state for {name}")
            getattr(self, name).copy_(value)
        self.fitted.fill_(bool(payload.get("fitted", True)))
