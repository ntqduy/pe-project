#!/usr/bin/env bash
# Shared plumbing for every wrapper under scripts/.
#
# These wrappers hold NO scientific logic. Each one names an experiment from
# configs/experiments.yaml and hands it to run.py, which owns the CLI. Anything a wrapper
# would have to decide -- which cohort, which backbone, which encoder checkpoint -- is a
# config variable, passed through as `--set`, so a shell script never becomes a second
# place where an experiment is defined.
#
# Environment (all optional):
#   DATASET             smoke_30 | test_500_sample | full_inspect
#                       default: the selected dataset wrapper's own profile; test_500_sample
#                       for every other stage
#   BACKBONE            ct_fm | ct_clip | totalfm             default: the config's own
#   ENCODER_SOURCE      pretrained | dapt | c0 | silver       default: the config's own
#   ENCODER_CHECKPOINT  explicit checkpoint for ENCODER_SOURCE (required when it is not
#                       the canonical run of that stage, e.g. another backbone's DAPT)
#   ENCODER_EXPERIMENT  the experiment id that produced ENCODER_CHECKPOINT
#   ACTION              show | plan | preflight | dry | run    default: run
#   GPUS                physical GPU ids, e.g. 0 or 0,1
#   SET                 extra config overrides, e.g. SET="training.epochs=1 seed=7"
#   RESUME=1 | OVERWRITE=1
#   Scoped stages only (dataset, segmentation, roi, silver, counterfactual):
#     MAX_CASES=N | MAX_REPORTS=N | PATIENT_ID=id | ALLOW_ALL=1
#
# A training stage has no case limit: its scope IS the dataset profile. Rehearse those on
# DATASET=test_500_sample, and use ACTION=dry / ACTION=preflight to check a run without
# starting it. SMOKE=1 additionally pins training.epochs=1.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Pipeline artifacts must live on the storage mount. This helper only validates a mount
# the user has already made; it never invokes gcsfuse or mounts anything itself.
# Direct `python run.py ...` invocations should source the same helper explicitly.
if [[ -z "${PE_CLOUD_ROOT:-}" && -f "$PROJECT_ROOT/scripts/use_gcs_storage.sh" ]]; then
  # shellcheck disable=SC1091 -- this is an intentional repository-local environment helper
  source "$PROJECT_ROOT/scripts/use_gcs_storage.sh"
fi
PYTHON="${PYTHON:-python}"

pe_die() { printf 'error: %s\n' "$*" >&2; exit 2; }

pe_run() {
  local scoped=0
  if [[ "${1:-}" == "--scoped" ]]; then scoped=1; shift; fi
  local experiment="${1:-}"
  [[ -n "$experiment" ]] || pe_die "pe_run needs an experiment name"
  shift || true

  local action="${ACTION:-run}"
  case "$action" in show|plan|preflight|dry|run) ;; *) pe_die "ACTION must be show, plan, preflight, dry or run" ;; esac

  local -a cmd=("$PYTHON" "$PROJECT_ROOT/run.py" "$action" "$experiment")
  local -a overrides=()

  local dataset="${DATASET:-}"
  if [[ -z "$dataset" ]]; then
    case "$experiment" in
      data.dataset.*) dataset="${experiment##*.}" ;;
      *) dataset="test_500_sample" ;;
    esac
  fi
  overrides+=("data.profile=${dataset}")
  [[ -n "${BACKBONE:-}" ]] && overrides+=("model.backbone=${BACKBONE}")
  [[ -n "${ENCODER_SOURCE:-}" ]] && overrides+=("encoder.init_source=${ENCODER_SOURCE}")
  [[ -n "${ENCODER_CHECKPOINT:-}" ]] && overrides+=("encoder.checkpoint=${ENCODER_CHECKPOINT}")
  [[ -n "${ENCODER_EXPERIMENT:-}" ]] && overrides+=("encoder.source_experiment=${ENCODER_EXPERIMENT}")
  [[ -n "${SMOKE:-}" ]] && overrides+=("training.epochs=1")
  if [[ -n "${SET:-}" ]]; then
    # shellcheck disable=SC2206 - word splitting is the documented interface
    local -a extra=(${SET})
    overrides+=("${extra[@]}")
  fi
  overrides+=("$@")

  if [[ "$action" == "show" || "$action" == "plan" ]]; then
    # These two only read the registry; they take no overrides.
    printf 'note: %s ignores DATASET/BACKBONE/ENCODER_SOURCE overrides.\n' "$action" >&2
    "${cmd[@]}"
    return
  fi

  local override
  for override in "${overrides[@]}"; do cmd+=(--set "$override"); done
  [[ -n "${GPUS:-}" ]] && cmd+=(--gpus "${GPUS}")
  [[ -n "${RESUME:-}" ]] && cmd+=(--resume)
  [[ -n "${OVERWRITE:-}" ]] && cmd+=(--overwrite)

  if (( scoped )); then
    if [[ -n "${PATIENT_ID:-}" ]]; then cmd+=(--patient-id "${PATIENT_ID}")
    elif [[ -n "${MAX_CASES:-}" ]]; then cmd+=(--max-cases "${MAX_CASES}")
    elif [[ -n "${MAX_REPORTS:-}" ]]; then cmd+=(--max-reports "${MAX_REPORTS}")
    elif [[ -n "${ALLOW_ALL:-}" ]]; then cmd+=(--allow-full)
    elif [[ "$action" == "run" ]]; then
      pe_die "$experiment needs an explicit scope: MAX_CASES=10 (smoke), PATIENT_ID=<id>, or ALLOW_ALL=1"
    fi
  else
    for variable in MAX_CASES MAX_REPORTS PATIENT_ID; do
      if [[ -n "${!variable:-}" ]]; then
        pe_die "$experiment is a training stage and has no per-case limit. Its scope is the
  dataset profile: rehearse with DATASET=test_500_sample, and use ACTION=dry or
  ACTION=preflight (optionally SMOKE=1) to check the run without training on everything."
      fi
    done
  fi

  printf '==> %s  [dataset=%s%s%s]\n' "$experiment" "$dataset" \
    "${BACKBONE:+ backbone=$BACKBONE}" "${ENCODER_SOURCE:+ encoder=$ENCODER_SOURCE}"
  "${cmd[@]}"
}
