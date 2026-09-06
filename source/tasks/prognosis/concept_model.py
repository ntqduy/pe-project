from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from source.components.adapters.organ import OrganAdapterBank
from source.components.anatomy import extract_anatomy_features
from source.components.encoders.image.base import BaseImageEncoder
from source.components.roi.feature_extractor import ROIFeatureExtractor
from source.concepts.model import ConceptBottleneck
from source.concepts.schema import ConceptSpec

from .heads import PrognosisHead


class ConceptBottleneckPrognosisModel(nn.Module):
    """Optional concept-bottleneck prognosis model (section 6).

    image -> organ-specific image representations -> clinical concepts -> concepts (+ raw
    clinical variables) -> prognosis. Kept entirely separate from the main ``PrognosisModel``:
    this is an optional, explicitly-opted-into experiment, never a requirement for the direct
    multimodal model. Every concept it predicts must already have a real native, validated-
    silver, or expert-reviewed target column - ``concepts.schema.enabled_concept_specs`` raises
    rather than let a proxy concept (e.g. an invented "embolic burden proxy") through silently.
    """

    def __init__(
        self,
        image_encoder: BaseImageEncoder,
        concepts: Mapping[str, ConceptSpec],
        clinical_encoder: nn.Module | None = None,
        regions: tuple[str, ...] = ("heart", "pa", "lung"),
        expert_dim: int = 128,
        concept_embedding_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.regions = regions
        self.roi = ROIFeatureExtractor(regions)
        self.organ_adapters = OrganAdapterBank(image_encoder.feature_dim, expert_dim, regions)
        pooled_dim = expert_dim * len(self.organ_adapters.feature_names)
        self.concept_bottleneck = ConceptBottleneck(
            pooled_dim, concepts, concept_embedding_dim, hidden_dim=hidden_dim
        )
        self.clinical_encoder = clinical_encoder
        fused_dim = concept_embedding_dim + (
            clinical_encoder.output_dim if clinical_encoder is not None else 0
        )
        self.head = PrognosisHead(fused_dim)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        volume: Tensor = batch["volume"]
        anatomy = extract_anatomy_features(
            self.image_encoder, self.roi, self.organ_adapters, volume, batch["masks"]
        )
        adapted, _ = anatomy.fusion_inputs(self.organ_adapters.feature_names)
        pooled = torch.cat([adapted[name] for name in self.organ_adapters.feature_names], dim=1)
        concept_output = self.concept_bottleneck(pooled)
        fused = concept_output["embedding"]
        if self.clinical_encoder is not None:
            clinical = self.clinical_encoder(batch["ehr"], batch.get("ehr_missing"))
            fused = torch.cat([fused, clinical], dim=1)
        return {
            "logits": self.head(fused),
            "concept_logits": concept_output["logits"],
            "concept_target_columns": self.concept_bottleneck.target_columns,
            "concept_target_specs": self.concept_bottleneck.target_specs,
            "features": fused,
            "anatomy_features": anatomy.as_dict(),
        }
