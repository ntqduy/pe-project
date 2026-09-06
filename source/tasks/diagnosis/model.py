from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from source.components.adapters.organ import OrganAdapterBank
from source.components.anatomy import extract_anatomy_features
from source.components.encoders.image.base import BaseImageEncoder
from source.components.fusion.base import FusionModule
from source.components.fusion.factory import build_fusion
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
        self.image_encoder = image_encoder
        self.regions = regions
        self.roi = ROIFeatureExtractor(regions)
        self.organ_adapters = OrganAdapterBank.from_config(
            image_encoder.feature_dim,
            expert_dim,
            regions,
            organ_adapter,
            default_hidden_dim=hidden_dim,
        )
        self.late_logit = str((fusion or {}).get("type", architecture)) == "late_logit"
        self.fusion: FusionModule = build_fusion(
            {"type": "concat_mlp"} if self.late_logit else fusion, self.organ_adapters.feature_names, self.organ_adapters.output_dim,
            fallback_type=architecture,
        )
        self.heads = DiagnosisHeads(self.fusion.output_dim, targets or DEFAULT_TARGETS)
        if self.late_logit:
            from source.components.fusion.late_logit import LateLogitFusion

            self.branch_heads = nn.ModuleDict({
                name: DiagnosisHeads(self.organ_adapters.output_dim, targets or DEFAULT_TARGETS)
                for name in self.organ_adapters.feature_names
            })
            self.logit_fusion = LateLogitFusion(
                self.organ_adapters.feature_names,
                learned=bool((fusion or {}).get("learned", False)), weights=(fusion or {}).get("weights"),
            )
            del self.fusion, self.heads
        self.auxiliary_target_mapping = normalize_organ_target_mapping(
            auxiliary_targets, default_source=auxiliary_default_source
        )
        missing_regions = sorted(set(self.auxiliary_target_mapping) - set(regions))
        if missing_regions:
            raise ValueError(
                "auxiliary heads require their pooled organ regions: " + ", ".join(missing_regions)
            )
        self.auxiliary_heads = nn.ModuleDict(
            {
                organ: DiagnosisHeads(
                    self.organ_adapters.output_dim,
                    {name: target.spec for name, target in organ_targets.items()},
                )
                for organ, organ_targets in self.auxiliary_target_mapping.items()
            }
        )

    def forward(self, volume: Tensor, masks: Mapping[str, Tensor]) -> dict[str, Any]:
        anatomy = extract_anatomy_features(
            self.image_encoder, self.roi, self.organ_adapters, volume, masks
        )
        adapted, adapted_present = anatomy.fusion_inputs(self.organ_adapters.feature_names)
        if self.late_logit:
            branches = {name: head(adapted[name]) for name, head in self.branch_heads.items()}
            logits = {}
            for target in next(iter(branches.values())):
                logits[target], routing = self.logit_fusion(
                    {name: values[target] for name, values in branches.items()}, adapted_present
                )
            fused = adapted["global"] if "global" in adapted else next(iter(adapted.values()))
        else:
            fused, routing = self.fusion(adapted, adapted_present)
            logits = self.heads(fused)
        auxiliary_logits = {
            organ: heads(adapted[organ]) for organ, heads in self.auxiliary_heads.items()
        }
        return {
            "logits": logits,
            "auxiliary_logits": auxiliary_logits,
            "auxiliary_target_mapping": self.auxiliary_target_mapping,
            "routing": routing,
            "features": fused,
            "anatomy_features": anatomy.as_dict(),
            "roi_present": {
                name: anatomy.available[name] for name in self.regions
            },
        }
