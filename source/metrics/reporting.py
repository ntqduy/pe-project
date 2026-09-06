from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DISCLAIMER = (
    "Structured reporting scaffold only. Populated fields point at where the corresponding "
    "evidence lives in this run's artifacts; they are not a claim of full STARD+AI/TRIPOD+AI "
    "compliance, which also requires narrative reporting this code does not generate."
)


def stard_ai_checklist(config: Mapping[str, Any], result_evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnosis-stage STARD+AI-oriented structured checklist (section 14)."""
    task = dict(config.get("task") or {})
    data = dict(config.get("data") or {})
    return {
        "checklist": "STARD-AI",
        "disclaimer": DISCLAIMER,
        "items": {
            "study_design": {"status": "see resolved_config.yaml", "dataset": data.get("dataset")},
            "reference_standard": {
                "status": "see result", "primary_target": result_evaluation.get("primary_target")
            },
            "participant_flow": {"status": "see lineage.json train/validation/test_patient_ids"},
            "ai_model_description": {
                "status": "see lineage.json", "architecture": task.get("architecture"),
                "regions": task.get("regions"), "input_counterfactual": task.get("input_counterfactual"),
            },
            "human_ai_comparison": {"status": "not applicable - no human reader arm in this pipeline"},
            "test_result_distribution": {
                "status": "see predictions.parquet and bootstrap_metrics.parquet if paired"
            },
            "diagnostic_accuracy": {
                "status": "see result.evaluation.metrics",
                "threshold_source": result_evaluation.get("threshold_source"),
            },
            "confidence_intervals": {
                "status": "see result.evaluation.metrics[*].ci_low/ci_high",
                "bootstrap": result_evaluation.get("bootstrap"),
            },
            "adverse_events": {"status": "not applicable - retrospective research model"},
        },
    }


def tripod_ai_checklist(config: Mapping[str, Any], result_evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Prognosis-stage TRIPOD+AI-oriented structured checklist (section 14)."""
    task = dict(config.get("task") or {})
    data = dict(config.get("data") or {})
    return {
        "checklist": "TRIPOD-AI",
        "disclaimer": DISCLAIMER,
        "items": {
            "source_of_data": {"status": "see resolved_config.yaml", "cohort": data.get("cohort")},
            "outcome": {
                "status": "see result", "primary_target": result_evaluation.get("primary_target")
            },
            "predictors": {
                "status": "see resolved_config.yaml", "modalities": task.get("modalities"),
                "ehr_columns": data.get("ehr_columns"), "pesi_columns": data.get("pesi_columns"),
            },
            "sample_size": {"status": "see lineage.json train/validation/test_patient_ids counts"},
            "missing_data": {
                "status": "see clinical preprocessing (train-fit only)",
                "missingness_indicators": task.get("ehr_include_missingness"),
            },
            "model_development": {
                "status": "see lineage.json", "fusion_type": config.get("fusion", {}).get("type"),
                "organ_adapter": config.get("organ_adapter"),
            },
            "model_performance": {
                "status": "see result.evaluation.metrics (AUROC/AUPRC/sensitivity/specificity/Brier/calibration)"
            },
            "confidence_intervals": {
                "status": "see result.evaluation.metrics[*].ci_low/ci_high",
                "bootstrap": result_evaluation.get("bootstrap"),
            },
            "risk_groups": {"status": "not implemented - report threshold-based classification only"},
        },
    }
