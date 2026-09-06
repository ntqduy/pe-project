from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .preprocessing import ClinicalPreprocessor


class ClinicalEncoder(nn.Module):
    """Small MLP over raw clinical values with optional missingness indicators."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        *,
        include_missingness: bool = True,
        preprocessing: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        options = dict(preprocessing or {})
        fit_split = str(options.get("fit_split", "train"))
        if fit_split != "train":
            raise ValueError("clinical.preprocessing.fit_split must be train")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.include_missingness = bool(include_missingness)
        self.preprocessor = ClinicalPreprocessor(
            self.input_dim,
            normalization=str(options.get("normalization", "zscore")),
            imputation=str(options.get("imputation", "mean")),
            constant_value=float(options.get("constant_value", 0.0)),
        )
        encoded_dim = self.input_dim * (2 if self.include_missingness else 1)
        self.network = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def fit_preprocessor(self, values: Tensor, *, split: str = "train") -> None:
        self.preprocessor.fit(values, split=split)

    def forward(self, values: Tensor, missing: Tensor | None = None) -> Tensor:
        normalized, missing_mask = self.preprocessor(values, missing)
        inputs = (
            __import__("torch").cat((normalized, missing_mask.to(normalized.dtype)), dim=1)
            if self.include_missingness
            else normalized
        )
        return self.network(inputs)
