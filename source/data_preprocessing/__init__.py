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
    -> EHR          ehr.build_ehr_readiness          strict pre-CTPA clinical readiness
    -> PESI         pesi.build_pesi_artifacts        approved-score gate + mapping audit
    -> manifests    manifests.build_manifests        ctpa / diagnosis / prognosis / reports
"""

from .adjudication import (
    PatientLabel,
    adjudicate_binary,
    mortality_outcome,
    patient_labels,
)
from .ehr import (
    EHR_FEATURE_COLUMNS,
    EhrArtifacts,
    EhrBuildError,
    EhrProfilesArtifacts,
    build_ehr_profiles,
    build_ehr_readiness,
)
from .filters import EligibilityReport, ExclusionLedger, apply_eligibility
from .integrity import VolumeCheck, check_volumes
from .leakage import SplitAudit, audit_split_integrity, load_excluded_patients
from .manifests import build_manifests
from .pesi import PESI_FEATURE_COLUMNS, PesiArtifacts, PesiBuildError, build_pesi_artifacts
from .pipeline import DatasetBuildError, build_dataset, dataset_output_root
from .sampling import SamplingPlan, sample_patients
from .sources import InspectSource, StudyRecord
from .volumes import (
    PREPROCESSING_IMPLEMENTATION,
    PatchCoordinate,
    PreprocessingSpec,
    patch_grid,
    preprocess_study,
    preprocess_volume,
    preprocess_volume_with_metadata,
    validate_cache_entry,
)

__all__ = [
    "DatasetBuildError",
    "EHR_FEATURE_COLUMNS",
    "EhrArtifacts",
    "EhrBuildError",
    "EhrProfilesArtifacts",
    "PESI_FEATURE_COLUMNS",
    "PesiArtifacts",
    "PesiBuildError",
    "PREPROCESSING_IMPLEMENTATION",
    "PatchCoordinate",
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
    "build_ehr_readiness",
    "build_ehr_profiles",
    "build_manifests",
    "build_pesi_artifacts",
    "check_volumes",
    "dataset_output_root",
    "load_excluded_patients",
    "mortality_outcome",
    "patient_labels",
    "preprocess_study",
    "preprocess_volume",
    "preprocess_volume_with_metadata",
    "sample_patients",
    "patch_grid",
    "validate_cache_entry",
]
