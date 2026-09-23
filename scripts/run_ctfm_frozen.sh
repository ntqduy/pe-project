#!/usr/bin/env bash
# Prepare, train, and evaluate CT-FM frozen + MLP runs without changing the official split.
#
# Examples:
#   PROFILE=full_inspect ACTION=all bash scripts/run_ctfm_frozen.sh
#   PROFILE=smoke_30 EPOCHS=1 ACTION=all bash scripts/run_ctfm_frozen.sh
#   PROFILE=full_inspect TASK=prognosis COHORT=pe GPUS=0,1 bash scripts/run_ctfm_frozen.sh
#   ACTION=preflight PROFILE=full_inspect TASK=diagnosis bash scripts/run_ctfm_frozen.sh
#
# The base dataset manifests must already exist. This script never calls the dataset builder
# and never creates a train/validation/test split.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-full_inspect}"
TASK="${TASK:-diagnosis}"
COHORT="${COHORT:-all}"
TARGET="${TARGET:-}"
ACTION="${ACTION:-all}"
PYTHON="${PYTHON:-python3}"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING="${EARLY_STOPPING:-10}"
NUM_WORKERS="${NUM_WORKERS:-}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS=1
fi
RAW_ROOT="${PE_RAW_INSPECT_ROOT:-/mnt/Stanford_INSPECT_dataset}"
if [[ -n "${PE_DERIVED_ROOT:-}" ]]; then
  DERIVED_ROOT="$PE_DERIVED_ROOT"
elif [[ -n "${PE_CLOUD_ROOT:-}" && -d "${PE_CLOUD_ROOT}/derived" ]]; then
  # This is the layout used by the mounted workspace in this project.
  DERIVED_ROOT="${PE_CLOUD_ROOT}/derived"
else
  DERIVED_ROOT="${PE_CLOUD_ROOT}/data/derived"
fi
DATASET_ROOT="${DERIVED_ROOT}/datasets/${PROFILE}"

case "$TASK" in
  diagnosis)
    CONFIG="${PROJECT_ROOT}/configs/runs/01_foundation/ct_fm_frozen_diagnosis.yaml"
    MANIFEST="ct_fm_frozen/manifests/diagnosis.csv"
    RUN_ID="DX_ctfm_frozen"
    ;;
  prognosis)
    case "$COHORT" in
      all|all_patient)
        CONFIG="${PROJECT_ROOT}/configs/runs/01_foundation/ct_fm_frozen_prognosis_all.yaml"
        MANIFEST="ct_fm_frozen/manifests/prognosis_all_patient.csv"
        RUN_ID="PR_ctfm_frozen_all"
        ;;
      pe|pe_positive|PE_positive)
        CONFIG="${PROJECT_ROOT}/configs/runs/01_foundation/ct_fm_frozen_prognosis_pe.yaml"
        MANIFEST="ct_fm_frozen/manifests/prognosis_pe_positive.csv"
        RUN_ID="PR_ctfm_frozen_pe"
        ;;
      *)
        printf 'error: COHORT must be all or pe for TASK=prognosis\n' >&2
        exit 2
        ;;
    esac
    ;;
  *)
    printf 'error: TASK must be diagnosis or prognosis\n' >&2
    exit 2
    ;;
esac

TARGET_ARGS=()
if [[ "$TASK" == "diagnosis" && -n "$TARGET" ]]; then
  printf 'error: TARGET is only valid for TASK=prognosis\n' >&2
  exit 2
fi
if [[ "$TASK" == "prognosis" && -n "$TARGET" ]]; then
  case "$TARGET" in
    1_month_mortality|6_month_mortality|12_month_mortality|1_month_readmission|6_month_readmission|12_month_readmission|12_month_PH) ;;
    *)
      printf 'error: unsupported prognosis TARGET: %s\n' "$TARGET" >&2
      printf '       valid: 1_month_mortality, 6_month_mortality, 12_month_mortality, 1_month_readmission, 6_month_readmission, 12_month_readmission, 12_month_PH\n' >&2
      exit 2
      ;;
  esac
  RUN_ID="${RUN_ID}_${TARGET}"
  TARGET_ARGS=(
    --set "task.primary_target=${TARGET}"
    --set "task.targets={${TARGET}: 1}"
    --set "data.label_columns=[${TARGET}]"
  )
fi

case "$ACTION" in
  prepare|train|evaluate|all|preflight|dry) ;;
  *) printf 'error: ACTION must be prepare, train, evaluate, all, preflight or dry\n' >&2; exit 2 ;;
esac

case "$EPOCHS" in
  ''|*[!0-9]*) printf 'error: EPOCHS must be a positive integer\n' >&2; exit 2 ;;
esac
(( EPOCHS >= 1 )) || { printf 'error: EPOCHS must be >= 1\n' >&2; exit 2; }
case "$EARLY_STOPPING" in
  ''|*[!0-9]*) printf 'error: EARLY_STOPPING must be a positive integer\n' >&2; exit 2 ;;
esac
(( EARLY_STOPPING >= 1 )) || { printf 'error: EARLY_STOPPING must be >= 1\n' >&2; exit 2; }

if [[ -z "${PE_CLOUD_ROOT:-}" && -z "${PE_DERIVED_ROOT:-}" ]]; then
  printf 'error: set PE_CLOUD_ROOT or PE_DERIVED_ROOT\n' >&2
  exit 2
fi
if [[ ! -f "${DATASET_ROOT}/manifests/diagnosis.csv" ]]; then
  printf 'error: existing official-split manifests not found under %s\n' "${DATASET_ROOT}" >&2
  printf '       build the selected dataset profile first; this script will not create a split\n' >&2
  exit 2
fi

if [[ "$ACTION" == "prepare" || "$ACTION" == "all" ]]; then
  if [[ -d "${DATASET_ROOT}/ct_fm_frozen" && -z "${OVERWRITE:-}" ]]; then
    printf '==> reusing existing CT-FM cache: %s\n' "${DATASET_ROOT}/ct_fm_frozen"
  else
    PREPARE_ARGS=()
    [[ -n "${OVERWRITE:-}" ]] && PREPARE_ARGS+=(--overwrite)
    "$PYTHON" "${PROJECT_ROOT}/tools/data/build_ctfm_cache.py" \
      --dataset-root "$DATASET_ROOT" \
      --raw-root "$RAW_ROOT" \
      --output-name ct_fm_frozen \
      --quiet \
      "${PREPARE_ARGS[@]}"
  fi
fi

TRAIN_ARGS=(
  --config "$CONFIG"
  --gpus "$GPUS"
  --set "data.profile=${PROFILE}"
  --set "data.manifest=${MANIFEST}"
  --set "experiment.id=${RUN_ID}"
  --set "training.epochs=${EPOCHS}"
  --set "training.early_stopping_patience=${EARLY_STOPPING}"
)
TRAIN_ARGS+=("${TARGET_ARGS[@]}")
[[ -n "${BATCH_SIZE:-}" ]] && TRAIN_ARGS+=(--set "training.batch_size=${BATCH_SIZE}")
[[ -n "${NUM_WORKERS:-}" ]] && TRAIN_ARGS+=(--set "compute.num_workers=${NUM_WORKERS}")
[[ -n "${OVERWRITE:-}" ]] && TRAIN_ARGS+=(--overwrite)

if [[ "$ACTION" == "preflight" ]]; then
  cd "$PROJECT_ROOT"
  PREFLIGHT_ARGS=(
    --config "$CONFIG" \
    --gpus "$GPUS" \
    --set "data.profile=${PROFILE}" \
    --set "data.manifest=${MANIFEST}" \
    --set "experiment.id=${RUN_ID}" \
    --set "training.epochs=${EPOCHS}" \
    --set "training.early_stopping_patience=${EARLY_STOPPING}"
  )
  PREFLIGHT_ARGS+=("${TARGET_ARGS[@]}")
  "$PYTHON" tools/preflight.py "${PREFLIGHT_ARGS[@]}"
  exit $?
fi

if [[ "$ACTION" == "dry" ]]; then
  cd "$PROJECT_ROOT"
  "$PYTHON" tools/launch.py "${TRAIN_ARGS[@]}" --dry-run
  exit $?
fi

# Capture the complete wrapper/launcher terminal stream without creating an incomplete
# experiment directory before OutputManager has performed its collision check.  The stream
# is copied into the per-epoch artifact bundle at the end of this script.
TERMINAL_LOG=""
if [[ "$ACTION" == "train" || "$ACTION" == "all" || "$ACTION" == "evaluate" ]]; then
  TERMINAL_LOG="$(mktemp /tmp/ctfm-terminal.XXXXXX.log)"
  trap 'if [[ -n "${TERMINAL_LOG:-}" ]]; then rm -f -- "$TERMINAL_LOG"; fi' EXIT
  exec > >(tee -a "$TERMINAL_LOG") 2>&1
fi

if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
  if [[ ! -f "${DATASET_ROOT}/${MANIFEST}" ]]; then
    printf 'error: CT-FM manifest not found: %s\n' "${DATASET_ROOT}/${MANIFEST}" >&2
    exit 2
  fi
  cd "$PROJECT_ROOT"
  # Use the launcher so GPUS=0,1 becomes a real torchrun/DDP job rather than a
  # single process that merely records two device IDs in the config.
  "$PYTHON" tools/launch.py --quiet "${TRAIN_ARGS[@]}"
fi

if [[ "$ACTION" == "evaluate" || "$ACTION" == "all" ]]; then
  if [[ ! -f "${DATASET_ROOT}/${MANIFEST}" ]]; then
    printf 'error: CT-FM manifest not found: %s\n' "${DATASET_ROOT}/${MANIFEST}" >&2
    exit 2
  fi
  cd "$PROJECT_ROOT"
  EVAL_ARGS=(
    --evaluate
    --config "$CONFIG"
    --gpus "$GPUS"
    --allow-full
    --set "data.profile=${PROFILE}"
    --set "data.manifest=${MANIFEST}"
    --set "experiment.id=${RUN_ID}"
    --set "training.epochs=${EPOCHS}"
    --set "training.early_stopping_patience=${EARLY_STOPPING}"
  )
  EVAL_ARGS+=("${TARGET_ARGS[@]}")
  [[ -n "${NUM_WORKERS:-}" ]] && EVAL_ARGS+=(--set "compute.num_workers=${NUM_WORKERS}")
  [[ -n "${OVERWRITE:-}" ]] && EVAL_ARGS+=(--overwrite)
  "$PYTHON" tools/launch.py --quiet "${EVAL_ARGS[@]}"
fi

if [[ "$ACTION" == "train" || "$ACTION" == "all" || "$ACTION" == "evaluate" ]]; then
  if [[ -n "${PE_CLOUD_PROJECT_ROOT:-}" ]]; then
    OUTPUT_ROOT="${PE_CLOUD_PROJECT_ROOT}/outputs"
  elif [[ -n "${PE_CLOUD_ROOT:-}" ]]; then
    OUTPUT_ROOT="${PE_CLOUD_ROOT}/pe-project/outputs"
  else
    OUTPUT_ROOT="<PE_CLOUD_PROJECT_ROOT>/outputs"
  fi
  FAMILY="diagnosis"
  [[ "$TASK" == "prognosis" ]] && FAMILY="prognosis"
  RUN_DIR="${OUTPUT_ROOT}/${FAMILY}/${RUN_ID}__ds_${PROFILE}"
  if [[ ! -d "$RUN_DIR" ]]; then
    RUN_DIR="$(find "${OUTPUT_ROOT}/${FAMILY}" -mindepth 1 -maxdepth 1 -type d -name "${RUN_ID}*" | sort | tail -n 1)"
  fi
  printf '\n==> completed task=%s cohort=%s profile=%s\n' "$TASK" "$COHORT" "$PROFILE"
  printf '    epochs=%s early_stopping_patience=%s gpus=%s\n' "$EPOCHS" "$EARLY_STOPPING" "$GPUS"
  printf '    output=%s\n' "$RUN_DIR"
  printf '    metrics=%s/result.json\n' "$RUN_DIR"
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
    EPOCH_DIR="$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' | sort | tail -n 1)"
    if [[ -n "$EPOCH_DIR" ]]; then
      printf '    log_file=%s/logs.txt\n' "$EPOCH_DIR"
    fi
  fi
fi
