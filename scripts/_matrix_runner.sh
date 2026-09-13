#!/usr/bin/env bash
# Shared launcher for the protocol matrix.  It owns only experiment selection and
# config overrides; run.py/tools/tasks/train_task.py remain the sole training engine.
#
# Public environment contract:
#   DATASET=smoke_30|test_500_sample|full_inspect
#   TASK=... LABEL=... COHORT=... EHR_PROFILE=...
#   WEIGHT_SOURCE=pretrained|dapt|alignment|rspect_multitask|rspect_single|silver_encoder|custom
#   WEIGHT_PATH=/path/to/checkpoint       (required for WEIGHT_SOURCE=custom)
#   WEIGHT_EXPERIMENT=upstream-run-id     (optional provenance override)
#   STRATEGY=...

set -euo pipefail

MATRIX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$MATRIX_SCRIPT_DIR/_lib.sh"

matrix_die() { printf 'error: %s\n' "$*" >&2; exit 2; }

matrix_token() {
  local name="${1:-}"
  local value="${2:-}"
  [[ -n "$value" && "$value" =~ ^[A-Za-z0-9_-]+$ ]] || \
    matrix_die "$name must contain only letters, numbers, '_' or '-' (got ${value@Q})"
}

matrix_weight_overrides() {
  local source="${WEIGHT_SOURCE:-alignment}"
  local init_source
  case "$source" in
    pretrained) init_source="pretrained" ;;
    dapt) init_source="dapt" ;;
    alignment) init_source="c0" ;;
    rspect_multitask) init_source="rspect_multitask" ;;
    rspect_single) init_source="rspect_single" ;;
    silver_encoder) init_source="silver" ;;
    custom) init_source="custom" ;;
    *)
      matrix_die "WEIGHT_SOURCE must be pretrained, dapt, alignment, rspect_multitask, rspect_single, silver_encoder or custom"
      ;;
  esac
  if [[ "$source" == "custom" && -z "${WEIGHT_PATH:-}" ]]; then
    matrix_die "WEIGHT_SOURCE=custom requires WEIGHT_PATH=/path/to/checkpoint"
  fi
  MATRIX_WEIGHT_SOURCE="$source"
  MATRIX_WEIGHT_OVERRIDES=(
    "encoder.init_source=${init_source}"
    "encoder.weight_source=${source}"
  )
  if [[ -n "${WEIGHT_PATH:-}" ]]; then
    MATRIX_WEIGHT_OVERRIDES+=("encoder.checkpoint=${WEIGHT_PATH}")
  fi
  if [[ -n "${WEIGHT_EXPERIMENT:-}" ]]; then
    MATRIX_WEIGHT_OVERRIDES+=("encoder.source_experiment=${WEIGHT_EXPERIMENT}")
  fi
}

# The dataset profile is an output axis for every stage that trains on INSPECT, but the
# canonical protocol run is the full cohort. Following the same rule as
# stamp_experiment_variant(), only a *deviation* from that default is stamped into the
# path. So the protocol run lands on the documented `weight_<source>/` directory and the
# canonical checkpoints named in components/encoders.yaml#sources stay resolvable, while a
# test_500_sample or smoke_30 rehearsal still gets a directory of its own.
MATRIX_DEFAULT_DATASET="full_inspect"

matrix_dataset_tag() {
  MATRIX_DATASET_TAG="${DATASET:-test_500_sample}"
  matrix_token DATASET "$MATRIX_DATASET_TAG"
  if [[ "$MATRIX_DATASET_TAG" == "$MATRIX_DEFAULT_DATASET" ]]; then
    MATRIX_DATASET_SUFFIX=""
  else
    MATRIX_DATASET_SUFFIX="__ds_${MATRIX_DATASET_TAG}"
  fi
}

matrix_run_rspect() {
  local mode="${RSPECT_MODE:-}"
  local experiment output_kind internal
  case "$mode" in
    multitask)
      experiment="repr.rspect.multitask"
      output_kind="multitask"
      internal="multitask"
      ;;
    single_pe)
      experiment="repr.rspect.single_task"
      output_kind="single_pe"
      internal="single"
      ;;
    *) matrix_die "RSPECT_MODE must be multitask or single_pe" ;;
  esac
  matrix_weight_overrides
  case "$MATRIX_WEIGHT_SOURCE" in
    pretrained|dapt|alignment|custom) ;;
    *) matrix_die "RSPECT may initialize only from pretrained, dapt, alignment or custom weights" ;;
  esac
  local -a overrides=(
    "experiment.id=DX_rspect_${internal}_weight_${MATRIX_WEIGHT_SOURCE}"
    "experiment.output_id=rspect/${output_kind}/weight_${MATRIX_WEIGHT_SOURCE}"
    "experiment.variant_stamp=false"
    "data.cohort=rspect_external"
    "${MATRIX_WEIGHT_OVERRIDES[@]}"
  )
  if [[ -n "${RSPECT_ROOT:-}" ]]; then
    overrides+=("data.root=${RSPECT_ROOT}")
  fi
  pe_run "$experiment" "${overrides[@]}" "$@"
}

# Diagnosis STRATEGY -> the experiment that actually implements it.
#
# This mapping is deliberately exhaustive rather than free-form. STRATEGY names the
# output directory, so accepting an unmapped value would produce a run labelled (say)
# `anatomy_soft_moe` that had in fact trained the plain global concat_mlp model. A new
# strategy is a new registry experiment, not a new string here.
matrix_diagnosis_strategy() {
  case "${1:-}/${2:-}" in
    multi_task/global) printf '%s\n' "diag.matrix.multitask" ;;
    single_task/global) printf '%s\n' "diag.matrix.single_task" ;;
    # Frozen-encoder linear probe: the fixed instrument for comparing WEIGHT_SOURCE.
    # Single-head by construction, so it has no multi_task form.
    single_task/probe) printf '%s\n' "probe.diag" ;;
    *) return 1 ;;
  esac
}

matrix_run_diagnosis() {
  local mode="${DIAGNOSIS_MODE:-}"
  local label="${LABEL:-}"
  local strategy="${STRATEGY:-global}"
  local experiment output_label internal
  matrix_token STRATEGY "$strategy"
  case "$mode" in
    multi_task)
      output_label="all_labels"
      internal="multitask"
      [[ -z "$label" || "$label" == "all_labels" ]] || \
        matrix_die "multi_task diagnosis trains all three labels; omit LABEL or set LABEL=all_labels"
      ;;
    single_task)
      case "$label" in pe_positive|pe_acute|pe_subsegmental) ;; *)
        matrix_die "single_task diagnosis LABEL must be pe_positive, pe_acute or pe_subsegmental" ;;
      esac
      output_label="$label"
      internal="single_${label}"
      ;;
    *) matrix_die "DIAGNOSIS_MODE must be multi_task or single_task" ;;
  esac
  experiment="$(matrix_diagnosis_strategy "$mode" "$strategy")" || \
    matrix_die "unknown diagnosis STRATEGY=${strategy@Q} for DIAGNOSIS_MODE=${mode}; \
supported: multi_task -> global; single_task -> global, probe"
  matrix_weight_overrides
  matrix_dataset_tag
  local -a overrides=(
    "experiment.id=DX_matrix_${internal}_ds_${MATRIX_DATASET_TAG}_weight_${MATRIX_WEIGHT_SOURCE}_${strategy}"
    "experiment.output_id=${mode}/${output_label}/weight_${MATRIX_WEIGHT_SOURCE}${MATRIX_DATASET_SUFFIX}/${strategy}"
    "experiment.variant_stamp=false"
    "data.cohort=all_eligible_ctpa"
    "${MATRIX_WEIGHT_OVERRIDES[@]}"
  )
  if [[ "$mode" == "single_task" ]]; then
    overrides+=(
      "data.label_columns=[${label}]"
      "diagnosis.primary_task=${label}"
      "diagnosis.tasks=[${label}]"
      "task.primary_target=${label}"
      "task.targets={${label}: 1}"
    )
  fi
  pe_run "$experiment" "${overrides[@]}" "$@"
}

matrix_run_silver() {
  local label="${SILVER_ENCODER_LABEL:-}"
  local experiment
  case "$label" in
    medgemma) experiment="repr.silver.medgemma" ;;
    rule_falcon) experiment="repr.silver.rule_falcon" ;;
    rule_falcon_medgemma) experiment="repr.silver.rule_falcon_medgemma" ;;
    *) matrix_die "SILVER_ENCODER_LABEL must be medgemma, rule_falcon or rule_falcon_medgemma" ;;
  esac
  # The protocol default is the RSPECT multitask encoder; callers may deliberately
  # compare another upstream stage with WEIGHT_SOURCE.
  local requested_weight="${WEIGHT_SOURCE:-rspect_multitask}"
  WEIGHT_SOURCE="$requested_weight" matrix_weight_overrides
  matrix_dataset_tag
  [[ "$MATRIX_WEIGHT_SOURCE" != "silver_encoder" ]] || \
    matrix_die "silver encoder adaptation cannot initialize from silver_encoder itself"
  local -a overrides=(
    "experiment.id=SE_silver_${label}_ds_${MATRIX_DATASET_TAG}_weight_${MATRIX_WEIGHT_SOURCE}"
    "experiment.output_id=silver_encoder/${label}/weight_${MATRIX_WEIGHT_SOURCE}${MATRIX_DATASET_SUFFIX}"
    "experiment.variant_stamp=false"
    "${MATRIX_WEIGHT_OVERRIDES[@]}"
  )
  pe_run "$experiment" "${overrides[@]}" "$@"
}

matrix_prognosis_strategy() {
  case "${1:-}" in
    image_only) printf '%s\n' "prog.image:image" ;;
    clinical_only) printf '%s\n' "prog.ehr:ehr" ;;
    pesi_only) printf '%s\n' "prog.pesi:pesi" ;;
    clinical_pesi) printf '%s\n' "prog.ehr_pesi:ehr,pesi" ;;
    image_clinical) printf '%s\n' "prog.image_ehr:image,ehr" ;;
    image_pesi) printf '%s\n' "prog.image_pesi:image,pesi" ;;
    image_clinical_pesi) printf '%s\n' "prog.image_ehr_pesi:image,ehr,pesi" ;;
    anatomy_concat) printf '%s\n' "prog.anatomy.concat:image,ehr,pesi" ;;
    anatomy_late_logit) printf '%s\n' "prog.anatomy.late:image,ehr,pesi" ;;
    anatomy_soft_moe) printf '%s\n' "prog.anatomy.moe:image,ehr,pesi" ;;
    global_late_logit) printf '%s\n' "prog.global.late:image,ehr,pesi" ;;
    global_soft_moe) printf '%s\n' "prog.global.moe:image,ehr,pesi" ;;
    probe) printf '%s\n' "probe.prog:image" ;;
    # ROI-only sufficiency students. Image-only by construction: the claim is that one
    # region alone carries the outcome signal, so the clinical branches must stay off.
    # Their reference row is image_only and their control is random_student.
    heart_student) printf '%s\n' "prog.heart_student:image" ;;
    pa_student) printf '%s\n' "prog.pa_student:image" ;;
    lung_student) printf '%s\n' "prog.lung_student:image" ;;
    random_student) printf '%s\n' "prog.random_student:image" ;;
    *) return 1 ;;
  esac
}

matrix_run_prognosis() {
  local cohort="${COHORT:-}"
  local ehr_profile="${EHR_PROFILE:-}"
  local task="${TASK:-}"
  local strategy="${STRATEGY:-}"
  local manifest
  case "$cohort" in
    all_patient|all_comers)
      cohort="all_comers"
      manifest="manifests/prognosis_all_patient.csv"
      ;;
    PE_positive|pe_positive_only)
      cohort="pe_positive_only"
      manifest="manifests/prognosis_pe_positive.csv"
      ;;
    *) matrix_die "COHORT must be all_comers or pe_positive_only (legacy aliases remain accepted)" ;;
  esac
  case "$ehr_profile" in
    EHR_0_h)
      MATRIX_EHR_COLUMNS="[ehr_index_time_missing,ehr_missing,ehr_has_crosswalk,ehr_events_prior,ehr_events_365d,ehr_events_30d,ehr_numeric_events_prior]"
      MATRIX_EHR_TABLE="clinical/ehr_features.csv"
      MATRIX_EHR_AVAILABILITY="has_ehr"
      ;;
    EHR_24_h)
      MATRIX_EHR_COLUMNS="[ehr_24h_index_time_missing,ehr_24h_missing,ehr_24h_has_crosswalk,ehr_24h_events_prior,ehr_24h_events_365d,ehr_24h_events_30d,ehr_24h_numeric_events_prior]"
      MATRIX_EHR_TABLE="clinical/EHR_24_h/ehr_features.csv"
      MATRIX_EHR_AVAILABILITY="has_ehr_24_h"
      ;;
    *) matrix_die "EHR_PROFILE must be EHR_0_h or EHR_24_h" ;;
  esac
  case "$task" in
    1_month_mortality|6_month_mortality|12_month_mortality|1_month_readmission|6_month_readmission|12_month_readmission|12_month_PH) ;;
    *) matrix_die "TASK must be one of the seven configured prognosis outcomes" ;;
  esac
  matrix_token STRATEGY "$strategy"
  local specification
  specification="$(matrix_prognosis_strategy "$strategy")" || \
    matrix_die "unknown prognosis STRATEGY=${strategy@Q}"
  local experiment="${specification%%:*}"
  local modalities="${specification#*:}"
  matrix_weight_overrides
  matrix_dataset_tag
  local -a overrides=(
    "experiment.id=PR_matrix_${cohort}_${ehr_profile}_${task}_ds_${MATRIX_DATASET_TAG}_weight_${MATRIX_WEIGHT_SOURCE}_${strategy}"
    "experiment.output_id=${cohort}/${ehr_profile}/${task}/weight_${MATRIX_WEIGHT_SOURCE}${MATRIX_DATASET_SUFFIX}/${strategy}"
    "experiment.variant_stamp=false"
    "data.cohort=${cohort}"
    "data.manifest=${manifest}"
    "data.label_columns=[${task}]"
    "data.ehr_profile=${ehr_profile}"
    "task.primary_target=${task}"
    "task.primary_label_index=0"
  )
  if [[ ",$modalities," == *,ehr,* ]]; then
    overrides+=(
      "data.ehr_columns=${MATRIX_EHR_COLUMNS}"
      "data.ehr_availability_column=${MATRIX_EHR_AVAILABILITY}"
      "task.ehr_input_dim=7"
      "supervision.ehr=${MATRIX_EHR_TABLE}"
    )
  fi
  if [[ ",$modalities," == *,pesi,* ]]; then
    overrides+=("data.pesi_availability_column=has_pesi")
  fi
  if [[ ",$modalities," == *,image,* ]]; then
    overrides+=("${MATRIX_WEIGHT_OVERRIDES[@]}")
  else
    overrides+=("lineage.weight_source=${MATRIX_WEIGHT_SOURCE}")
    printf 'note: WEIGHT_SOURCE=%s is recorded but does not affect tabular-only strategy=%s.\n' \
      "$MATRIX_WEIGHT_SOURCE" "$strategy" >&2
  fi
  pe_run "$experiment" "${overrides[@]}" "$@"
}
