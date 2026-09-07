"""Reusable INSPECT data-preprocessing logic.

Everything in this package treats the raw Stanford INSPECT release as READ-ONLY. It
reads the official metadata tables, applies one shared eligibility/QC contract, and
writes derived cohort manifests under the derived data root.

The scientific logic lives here so that every dataset profile
(``source/dataset/profiles/*.yaml``) shares one implementation: ``test_500_sample``
and ``full_inspect`` differ only by the sampling block, never by preprocessing code.

Stage order (``pipeline.build_dataset``):

    raw tables      sources.InspectSource
    -> eligibility  filters.apply_eligibility        exclusion ledger, noise removal
    -> integrity    integrity.check_volumes          corrupted / missing CT
    -> adjudication adjudication.patient_labels      per-patient label reconciliation
    -> sampling     sampling.sample_patients         patient-level, inside official split
    -> leakage      leakage.audit_split_integrity    official split + leakage guards
    -> manifests    manifests.build_manifests        ctpa / diagnosis / prognosis / reports
"""

from .adjudication import (
    PatientLabel,
    adjudicate_binary,
    mortality_outcome,
    patient_labels,
)
from .filters import EligibilityReport, ExclusionLedger, apply_eligibility
from .integrity import VolumeCheck, check_volumes
from .leakage import SplitAudit, audit_split_integrity, load_excluded_patients
from .manifests import build_manifests
from .pipeline import DatasetBuildError, build_dataset, dataset_output_root
from .sampling import SamplingPlan, sample_patients
from .sources import InspectSource, StudyRecord
from .volumes import PreprocessingSpec, preprocess_study, preprocess_volume

__all__ = [
    "DatasetBuildError",
    "EligibilityReport",
    "ExclusionLedger",
    "InspectSource",
    "PatientLabel",
    "PreprocessingSpec",
    "SamplingPlan",
    "SplitAudit",
    "StudyRecord",
    "VolumeCheck",
    "adjudicate_binary",
    "apply_eligibility",
    "audit_split_integrity",
    "build_dataset",
    "build_manifests",
    "check_volumes",
    "dataset_output_root",
    "load_excluded_patients",
    "mortality_outcome",
    "patient_labels",
    "preprocess_study",
    "preprocess_volume",
    "sample_patients",
]
