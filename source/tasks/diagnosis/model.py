"""Image-only PE diagnosis: shared CTPA encoder -> organ adapters -> fusion -> heads.

The architecture is deliberately four independent, swappable pieces, in this order:

    1. shared encoder   one forward pass over the whole CTPA volume
                        (source/components/encoders/image/*)
    2. organ adapters   the same feature map pooled into global / heart / pa / lung
                        branches, each with its own adapter
                        (source/components/roi + source/components/adapters/organ.py)
    3. prediction head  a linear head bank per target (heads.py)
    4. fusion           how branches become one decision (source/components/fusion/*)

Only step 4 changes between the ``concat`` / ``late_logit`` / ``soft_moe`` arms, and it is
selected purely by ``fusion.type`` in the config. Two mutually exclusive prediction paths
exist and exactly one is built:

    feature fusion (concat_mlp, soft_moe)   branches -> fusion -> one shared head bank
    late-logit fusion                       branches -> one head bank per branch ->
                                            masked, renormalized weighted logit average

Diagnosis is image-only by design: no EHR and no PESI ever enter this model.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from source.components.adapters.organ import OrganAdapterBank
from source.components.anatomy import extract_anatomy_features
from source.components.encoders.image.base import BaseImageEncoder
from source.components.fusion.factory import LATE_LOGIT, build_fusion, is_late_logit
from source.components.fusion.late_logit import LateLogitFusion
from source.components.roi.feature_extractor import ROIFeatureExtractor

from .heads import DiagnosisHeads
from .organ_targets import normalize_organ_target_mapping

DEFAULT_TARGETS = {
    "pe_present": 1,
    "acuity": 4,
    "central": 1,
    "lobar": 1,
    "segmental": 1,
    "subsegmental": 1,
    "saddle": 1,
    "rv_enlargement": 1,
    "rv_lv_abnormal": 1,
    "septal_bowing": 1,
    "reflux": 1,
    "pleural_effusion": 1,
    "pericardial_effusion": 1,
    "chronic_lung_disease": 1,
}


class DiagnosisModel(nn.Module):
    def __init__(
        self,
        image_encoder: BaseImageEncoder,
        targets: Mapping[str, int] | None = None,
        regions: tuple[str, ...] = ("heart", "pa", "lung"),
        expert_dim: int = 128,
        hidden_dim: int = 256,
        architecture: str = "soft_moe",
        organ_adapter: Mapping[str, Any] | None = None,
        fusion: Mapping[str, Any] | None = None,
        auxiliary_targets: Mapping[str, Any] | None = None,
        auxiliary_default_source: str = "native",
    ):
        super().__init__()
        options = dict(fusion or {})
        self.image_encoder = image_encoder
        self.regions = tuple(regions)
        self.targets = dict(targets or DEFAULT_TARGETS)
        self.roi = ROIFeatureExtractor(self.regions)
        self.organ_adapters = OrganAdapterBank.from_config(
            image_encoder.feature_dim,
            expert_dim,
            self.regions,
            organ_adapter,
            default_hidden_dim=hidden_dim,
        )
        self.branch_names = self.organ_adapters.feature_names
        branch_dim = self.organ_adapters.output_dim

        # Exactly one of the two prediction paths is constructed. `fusion`/`heads` are the
        # module names counterfactual runs list in lineage.transfer_modules, so they keep
        # those names on the feature-fusion path.
        self.fusion_type = str(options.get("type") or architecture).lower()
        if is_late_logit(self.fusion_type):
            self.branch_heads = nn.ModuleDict(
                {name: DiagnosisHeads(branch_dim, self.targets) for name in self.branch_names}
            )
            self.logit_fusion = LateLogitFusion(
                self.branch_names,
                learned=bool(options.get("learned", False)),
                weights=options.get("weights"),
            )
            self.fusion = None
            self.heads = None
        else:
            self.fusion = build_fusion(
                fusion, self.branch_names, branch_dim, fallback_type=architecture
            )
            self.heads = DiagnosisHeads(self.fusion.output_dim, self.targets)
            self.branch_heads = None
            self.logit_fusion = None

        self.auxiliary_target_mapping = normalize_organ_target_mapping(
            auxiliary_targets, default_source=auxiliary_default_source
        )
        missing_regions = sorted(set(self.auxiliary_target_mapping) - set(self.regions))
        if missing_regions:
            raise ValueError(
                "auxiliary heads require their pooled organ regions: " + ", ".join(missing_regions)
            )
        self.auxiliary_heads = nn.ModuleDict(
            {
                organ: DiagnosisHeads(
                    branch_dim, {name: target.spec for name, target in organ_targets.items()}
                )
                for organ, organ_targets in self.auxiliary_target_mapping.items()
            }
        )

    def predict(
        self, branch_features: Mapping[str, Tensor], branch_present: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor | None, Tensor]:
        """Turn adapted branch features into target logits, routing weights and features."""
        if self.logit_fusion is None:
            fused, routing = self.fusion(branch_features, branch_present)
            return self.heads(fused), routing, fused
        branch_logits = {name: head(branch_features[name]) for name, head in self.branch_heads.items()}
        logits: dict[str, Tensor] = {}
        routing: Tensor | None = None
        for target in self.targets:
            logits[target], routing = self.logit_fusion(
                {name: values[target] for name, values in branch_logits.items()}, branch_present
            )
        # Routing is identical for every target (it depends only on branch availability and
        # the branch weights), so reporting the last one is exact, not an approximation.
        fused = torch.cat([branch_features[name] for name in self.branch_names], dim=1)
        return logits, routing, fused

    def forward(self, volume: Tensor, masks: Mapping[str, Tensor]) -> dict[str, Any]:
        anatomy = extract_anatomy_features(
            self.image_encoder, self.roi, self.organ_adapters, volume, masks
        )
        adapted, adapted_present = anatomy.fusion_inputs(self.branch_names)
        logits, routing, fused = self.predict(adapted, adapted_present)
        auxiliary_logits = {
            organ: heads(adapted[organ]) for organ, heads in self.auxiliary_heads.items()
        }
        return {
            "logits": logits,
            "auxiliary_logits": auxiliary_logits,
            "auxiliary_target_mapping": self.auxiliary_target_mapping,
            "routing": routing,
            "features": fused,
            "branch_features": adapted,
            "anatomy_features": anatomy.as_dict(),
            "roi_present": {name: anatomy.available[name] for name in self.regions},
        }


__all__ = ["DEFAULT_TARGETS", "LATE_LOGIT", "DiagnosisModel"]
