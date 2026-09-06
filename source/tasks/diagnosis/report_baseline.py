from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .heads import DiagnosisHeads
from .model import DEFAULT_TARGETS


class ReportOnlyDiagnosisModel(nn.Module):
    """Report-embedding-only NLP baseline for the diagnosis comparison suite (DX_REPORT).

    Consumes precomputed report embeddings under the same ``alignment.report_embedding_columns``
    convention already used by the image-report alignment stage, instead of loading a second, separate
    text-encoder stack. This keeps the baseline real (it trains and evaluates on the same report
    representations used elsewhere) while isolating it from the image-centered diagnosis models -
    this model never sees the CTPA volume, and the image-centered models never see report text.
    """

    def __init__(self, report_dim: int, targets: Mapping[str, int] | None = None, hidden_dim: int = 256):
        super().__init__()
        if report_dim < 1:
            raise ValueError(
                "DX_REPORT requires alignment.report_embedding_columns to be a non-empty list"
            )
        self.encoder = nn.Sequential(
            nn.Linear(report_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.heads = DiagnosisHeads(hidden_dim, targets or DEFAULT_TARGETS)

    def forward(self, report_embedding: Tensor) -> dict[str, Any]:
        features = self.encoder(report_embedding)
        return {"logits": self.heads(features), "routing": None, "features": features, "roi_present": {}}
