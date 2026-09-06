from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from source.components.encoders.ehr import EHREncoder
from source.components.encoders.image.registry import build_image_encoder
from source.components.encoders.pesi import PESIEncoder
from source.components.peft.freeze import apply_peft
from source.tasks.contour.model import ContourModel
from source.tasks.diagnosis.model import DEFAULT_TARGETS, DiagnosisModel
from source.tasks.prognosis.model import PrognosisModel


def build_task_model(config: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    experiment = dict(config.get("experiment") or {})
    stage = str(experiment.get("stage"))
    model_config = dict(config.get("model") or {})
    model_config["data_mode"] = str((config.get("data") or {}).get("mode"))
    task = dict(config.get("task") or {})
    if stage in {"ablation", "counterfactual", "roi_student"}:
        stage = str(task.get("base_stage") or "")
        if stage not in {"diagnosis", "prognosis", "contour"}:
            raise ValueError(
                f"{experiment.get('stage')} task.base_stage must be diagnosis, prognosis, or contour"
            )
    peft_report: dict[str, Any] = {"method": "none", "modified_modules": []}
    if (config.get("probe") or {}).get("enabled"):
        from source.tasks.representation_probe import RepresentationProbe

        if stage not in {"diagnosis", "prognosis"}:
            raise ValueError("representation probes support diagnosis and prognosis")
        encoder = build_image_encoder(model_config)
        return RepresentationProbe(encoder, stage), {"method": "frozen", "modified_modules": []}
    if stage == "diagnosis":
        targets = task.get("targets") or DEFAULT_TARGETS
        if isinstance(targets, list):
            targets = {name: 1 for name in targets}
        if str(task.get("architecture", "soft_moe")) == "report_only":
            from source.tasks.diagnosis.report_baseline import ReportOnlyDiagnosisModel

            report_dim = len((config.get("alignment") or {}).get("report_embedding_columns") or ())
            model = ReportOnlyDiagnosisModel(
                report_dim, targets=targets, hidden_dim=int(task.get("hidden_dim", 256))
            )
            return model, peft_report
        encoder = build_image_encoder(model_config)
        peft_report = apply_peft(encoder, dict(config.get("peft") or {"method": "full"}))
        model = DiagnosisModel(
            encoder,
            targets=targets,
            regions=tuple(task.get("regions", ("heart", "pa", "lung"))),
            expert_dim=int(task.get("expert_dim", 128)),
            hidden_dim=int(task.get("hidden_dim", 256)),
            architecture=str(task.get("architecture", "soft_moe")),
            organ_adapter=config.get("organ_adapter"),
            fusion=config.get("fusion"),
            auxiliary_targets=task.get("auxiliary_targets"),
            auxiliary_default_source=str(task.get("auxiliary_default_source", "native")),
        )
    elif stage == "prognosis":
        from source.concepts.schema import enabled_concept_specs

        concepts = enabled_concept_specs(config.get("concept_bottleneck"))
        if concepts:
            from source.tasks.prognosis.concept_model import ConceptBottleneckPrognosisModel

            image = build_image_encoder(model_config)
            peft_report = apply_peft(image, dict(config.get("peft") or {"method": "full"}))
            clinical = (
                EHREncoder(
                    int(task.get("ehr_input_dim", 1)),
                    int(task.get("ehr_hidden_dim", 64)),
                    int(task.get("ehr_output_dim", 64)),
                    include_missingness=bool(task.get("ehr_include_missingness", True)),
                    preprocessing=dict(config.get("clinical_preprocessing") or {}),
                )
                if "ehr" in set(task.get("modalities") or ())
                else None
            )
            model = ConceptBottleneckPrognosisModel(
                image,
                concepts,
                clinical_encoder=clinical,
                regions=tuple(task.get("regions", ("heart", "pa", "lung"))),
                expert_dim=int(task.get("expert_dim", 128)),
                concept_embedding_dim=int(task.get("concept_embedding_dim", 64)),
                hidden_dim=int(task.get("hidden_dim", 256)),
            )
            return model, peft_report
        modalities = set(task.get("modalities") or ("image", "ehr", "pesi"))
        image = build_image_encoder(model_config) if "image" in modalities else None
        if image is not None:
            peft_report = apply_peft(image, dict(config.get("peft") or {"method": "full"}))
        ehr = EHREncoder(
            int(task.get("ehr_input_dim", 1)),
            int(task.get("ehr_hidden_dim", 64)),
            int(task.get("ehr_output_dim", 64)),
            include_missingness=bool(task.get("ehr_include_missingness", True)),
            preprocessing=dict(config.get("clinical_preprocessing") or {}),
        ) if "ehr" in modalities else None
        pesi = PESIEncoder(
            int(task.get("pesi_input_dim", 2)),
            int(task.get("pesi_hidden_dim", 16)),
            int(task.get("pesi_output_dim", 32)),
        ) if "pesi" in modalities else None
        model = PrognosisModel(
            image,
            ehr,
            pesi,
            regions=tuple(task.get("regions", ("heart", "pa", "lung"))),
            expert_dim=int(task.get("expert_dim", 128)),
            hidden_dim=int(task.get("hidden_dim", 256)),
            architecture=str(task.get("architecture", "soft_moe")),
            organ_adapter=config.get("organ_adapter"),
            fusion=config.get("fusion"),
        )
    elif stage == "contour":
        encoder = build_image_encoder(model_config)
        peft_report = apply_peft(encoder, dict(config.get("peft") or {"method": "full"}))
        model = ContourModel(
            encoder,
            regions=int(task.get("output_regions", 1)),
            decoder_channels=int(task.get("decoder_channels", 64)),
        )
    else:
        raise ValueError(f"task model factory does not support stage={stage!r}")
    return model, peft_report
