from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from source.components.adapters.organ import OrganAdapterBank, build_organ_adapter
from source.components.anatomy import extract_anatomy_features
from source.components.encoders.ehr import EHREncoder
from source.components.encoders.image.base import BaseImageEncoder
from source.components.encoders.pesi import PESIEncoder
from source.components.fusion.base import FusionModule
from source.components.fusion.factory import build_fusion
from source.components.roi.feature_extractor import ROIFeatureExtractor

from .heads import PrognosisHead


class PrognosisModel(nn.Module):
    def __init__(
        self,
        image_encoder: BaseImageEncoder | None,
        ehr_encoder: EHREncoder | None = None,
        pesi_encoder: PESIEncoder | None = None,
        regions: tuple[str, ...] = ("heart", "pa", "lung"),
        expert_dim: int = 128,
        hidden_dim: int = 256,
        architecture: str = "soft_moe",
        organ_adapter: Mapping[str, Any] | None = None,
        fusion: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if image_encoder is None and ehr_encoder is None and pesi_encoder is None:
            raise ValueError("prognosis requires at least one enabled modality")
        self.image_encoder = image_encoder
        self.ehr_encoder = ehr_encoder
        self.pesi_encoder = pesi_encoder
        self.regions = regions

        adapter_options = dict(organ_adapter or {})
        adapter_type = str(adapter_options.get("type", "bottleneck_mlp"))
        adapter_dropout = float(adapter_options.get("dropout", 0.1))
        adapter_rank = int(adapter_options.get("rank", 8))
        adapter_alpha = float(adapter_options.get("alpha", 16.0))

        feature_names: list[str] = []
        if image_encoder is not None:
            self.roi = ROIFeatureExtractor(regions)
            self.organ_adapters = OrganAdapterBank.from_config(
                image_encoder.feature_dim,
                expert_dim,
                regions,
                organ_adapter,
                default_hidden_dim=hidden_dim,
            )
            feature_names.extend(self.organ_adapters.feature_names)
        else:
            self.roi = None
            self.organ_adapters = None
        if ehr_encoder is not None:
            self.ehr_adapter = build_organ_adapter(
                adapter_type,
                ehr_encoder.output_dim,
                hidden_dim,
                expert_dim,
                dropout=adapter_dropout,
                rank=adapter_rank,
                alpha=adapter_alpha,
            )
            feature_names.append("ehr")
        if pesi_encoder is not None:
            self.pesi_adapter = build_organ_adapter(
                adapter_type,
                pesi_encoder.output_dim,
                hidden_dim,
                expert_dim,
                dropout=adapter_dropout,
                rank=adapter_rank,
                alpha=adapter_alpha,
            )
            feature_names.append("pesi")

        # All fusion inputs must share one width. The EHR/PESI adapters above always emit
        # expert_dim; the image organ-adapter bank only diverges from it when
        # organ_adapter.enabled=False, in which case image_encoder.feature_dim must equal
        # expert_dim for a multimodal (image + ehr/pesi) experiment to be shape-consistent.
        fusion_feature_dim = self.organ_adapters.output_dim if self.organ_adapters is not None else expert_dim
        self.late_logit = str((fusion or {}).get("type", architecture)) == "late_logit"
        self.fusion: FusionModule = build_fusion(
            {"type": "concat_mlp"} if self.late_logit else fusion, feature_names, fusion_feature_dim, fallback_type=architecture
        )
        self.head = PrognosisHead(self.fusion.output_dim)
        if self.late_logit:
            from source.components.fusion.late_logit import LateLogitFusion

            self.branch_heads = nn.ModuleDict({name: PrognosisHead(fusion_feature_dim) for name in feature_names})
            self.logit_fusion = LateLogitFusion(
                feature_names, learned=bool((fusion or {}).get("learned", False)),
                weights=(fusion or {}).get("weights"),
            )
            del self.fusion, self.head

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        features: dict[str, Tensor] = {}
        availability: dict[str, Tensor] = {}
        roi_present: dict[str, Tensor] = {}
        reference: Tensor | None = None
        if self.image_encoder is not None:
            volume = batch["volume"]
            reference = volume
            anatomy = extract_anatomy_features(
                self.image_encoder, self.roi, self.organ_adapters, volume, batch["masks"]
            )
            adapted_image, adapted_present = anatomy.fusion_inputs(
                self.organ_adapters.feature_names
            )
            roi_present = {
                name: anatomy.available[name] for name in self.regions
            }
            features.update(adapted_image)
            availability.update(adapted_present)
        if self.ehr_encoder is not None:
            values = batch["ehr"]
            reference = values
            encoded = self.ehr_encoder(values, batch.get("ehr_missing"))
            features["ehr"] = self.ehr_adapter(encoded)
            availability["ehr"] = ~torch.isnan(values).all(dim=1)
        if self.pesi_encoder is not None:
            scores = batch["pesi"]
            reference = scores
            encoded = self.pesi_encoder(scores, batch.get("pesi_missing"))
            features["pesi"] = self.pesi_adapter(encoded)
            availability["pesi"] = ~torch.isnan(scores).all(dim=1)
        if reference is None:
            raise RuntimeError("no prognosis modality was evaluated")
        if self.late_logit:
            logits, routing = self.logit_fusion(
                {name: head(features[name]) for name, head in self.branch_heads.items()}, availability
            )
            fused = next(iter(features.values()))
        else:
            fused, routing = self.fusion(features, availability)
            logits = self.head(fused)
        output = {
            "logits": logits,
            "routing": routing,
            "features": fused,
            "roi_present": roi_present,
        }
        if self.image_encoder is not None:
            output["anatomy_features"] = anatomy.as_dict()
        return output
