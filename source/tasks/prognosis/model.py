"""30-day mortality prognosis: the diagnosis imaging stack plus late clinical fusion.

The imaging half is deliberately the same four pieces as diagnosis -- shared CTPA encoder,
organ adapters, prediction head, fusion -- so a prognosis result can be compared with a
diagnosis result without the architecture being a confound. Prognosis then adds two more
branches *at the fusion stage only*:

    ehr    the leakage-checked pre-CTPA clinical feature vector (source/clinical/encoder.py)
    pesi   the PESI / sPESI severity scores      (source/components/encoders/pesi.py)

Every branch -- global, heart, pa, lung, ehr, pesi -- is adapted to one common width and
then fused exactly once, at the end. `task.modalities` selects which branches exist, and
`fusion.type` selects how they are combined:

    feature fusion (concat_mlp, soft_moe)   branches -> fusion -> one head
    late-logit fusion                       branches -> one head per branch -> masked,
                                            renormalized weighted logit average

A branch whose input is missing for a patient (no mask, no clinical record, no PESI score)
is masked out and, under late-logit and Soft-MoE fusion, renormalized away rather than fed
as zeros.
"""
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
from source.components.fusion.factory import LATE_LOGIT, build_fusion, is_late_logit
from source.components.fusion.late_logit import LateLogitFusion
from source.components.roi.feature_extractor import ROIFeatureExtractor

from .heads import PrognosisHead

CLINICAL_BRANCHES = ("ehr", "pesi")


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
        options = dict(fusion or {})
        adapter_options = dict(organ_adapter or {})
        self.image_encoder = image_encoder
        self.ehr_encoder = ehr_encoder
        self.pesi_encoder = pesi_encoder
        self.regions = tuple(regions)

        branch_names: list[str] = []
        if image_encoder is not None:
            self.roi = ROIFeatureExtractor(self.regions)
            self.organ_adapters = OrganAdapterBank.from_config(
                image_encoder.feature_dim,
                expert_dim,
                self.regions,
                organ_adapter,
                default_hidden_dim=hidden_dim,
            )
            branch_names.extend(self.organ_adapters.feature_names)
        else:
            self.roi = None
            self.organ_adapters = None

        # All fusion inputs must share one width. The clinical adapters below always emit
        # expert_dim; the image organ-adapter bank only diverges from it when
        # organ_adapter.enabled=False, in which case image_encoder.feature_dim must equal
        # expert_dim for a multimodal (image + ehr/pesi) experiment to be shape-consistent.
        branch_dim = (
            self.organ_adapters.output_dim if self.organ_adapters is not None else expert_dim
        )
        self.clinical_adapters = nn.ModuleDict()
        for name, encoder in (("ehr", ehr_encoder), ("pesi", pesi_encoder)):
            if encoder is None:
                continue
            self.clinical_adapters[name] = build_organ_adapter(
                str(adapter_options.get("type", "bottleneck_mlp")),
                encoder.output_dim,
                hidden_dim,
                branch_dim,
                dropout=float(adapter_options.get("dropout", 0.1)),
                rank=int(adapter_options.get("rank", 8)),
                alpha=float(adapter_options.get("alpha", 16.0)),
            )
            branch_names.append(name)
        self.branch_names = tuple(branch_names)

        # Exactly one prediction path is built; `fusion`/`head` keep their names on the
        # feature-fusion path because checkpoint transfer lists them by name.
        self.fusion_type = str(options.get("type") or architecture).lower()
        if is_late_logit(self.fusion_type):
            self.branch_heads = nn.ModuleDict(
                {name: PrognosisHead(branch_dim) for name in self.branch_names}
            )
            self.logit_fusion = LateLogitFusion(
                self.branch_names,
                learned=bool(options.get("learned", False)),
                weights=options.get("weights"),
            )
            self.fusion = None
            self.head = None
        else:
            self.fusion = build_fusion(
                fusion, self.branch_names, branch_dim, fallback_type=architecture
            )
            self.head = PrognosisHead(self.fusion.output_dim)
            self.branch_heads = None
            self.logit_fusion = None

    def predict(
        self, branch_features: Mapping[str, Tensor], branch_present: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Turn adapted branch features into the mortality logit, routing and features."""
        if self.logit_fusion is None:
            fused, routing = self.fusion(branch_features, branch_present)
            return self.head(fused), routing, fused
        branch_logits = {
            name: head(branch_features[name]) for name, head in self.branch_heads.items()
        }
        logits, routing = self.logit_fusion(branch_logits, branch_present)
        fused = torch.cat([branch_features[name] for name in self.branch_names], dim=1)
        return logits, routing, fused

    def _clinical_branch(
        self,
        name: str,
        values: Tensor,
        encoder: nn.Module,
        missing: Tensor | None,
        available: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        encoded = encoder(values, missing)
        inferred_available = ~torch.isnan(values).all(dim=1)
        if available is not None:
            explicit = available.to(values.device).bool().reshape(-1)
            if explicit.shape != inferred_available.shape:
                raise ValueError(
                    f"{name}_available must have one value per batch row, got {tuple(explicit.shape)}"
                )
            inferred_available = inferred_available & explicit
        return self.clinical_adapters[name](encoded), inferred_available

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        features: dict[str, Tensor] = {}
        availability: dict[str, Tensor] = {}
        roi_present: dict[str, Tensor] = {}
        anatomy = None
        if self.image_encoder is not None:
            anatomy = extract_anatomy_features(
                self.image_encoder, self.roi, self.organ_adapters, batch["volume"], batch["masks"]
            )
            adapted_image, adapted_present = anatomy.fusion_inputs(
                self.organ_adapters.feature_names
            )
            roi_present = {name: anatomy.available[name] for name in self.regions}
            features.update(adapted_image)
            availability.update(adapted_present)
        if self.ehr_encoder is not None:
            features["ehr"], availability["ehr"] = self._clinical_branch(
                "ehr",
                batch["ehr"],
                self.ehr_encoder,
                batch.get("ehr_missing"),
                batch.get("ehr_available"),
            )
        if self.pesi_encoder is not None:
            features["pesi"], availability["pesi"] = self._clinical_branch(
                "pesi",
                batch["pesi"],
                self.pesi_encoder,
                batch.get("pesi_missing"),
                batch.get("pesi_available"),
            )
        if not features:
            raise RuntimeError("no prognosis modality was evaluated")
        logits, routing, fused = self.predict(features, availability)
        output: dict[str, Any] = {
            "logits": logits,
            "routing": routing,
            "features": fused,
            "branch_features": features,
            "roi_present": roi_present,
        }
        if anatomy is not None:
            output["anatomy_features"] = anatomy.as_dict()
        return output


__all__ = ["CLINICAL_BRANCHES", "LATE_LOGIT", "PrognosisModel"]
