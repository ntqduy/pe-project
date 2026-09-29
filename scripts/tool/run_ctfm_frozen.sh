#!/usr/bin/env bash
# Prepare, train, and evaluate CT-FM frozen + MLP runs without changing the official split.
#
# Examples:
#   PROFILE=full_inspect ACTION=all bash scripts/tool/run_ctfm_frozen.sh
#   PROFILE=smoke_30 EPOCHS=1 ACTION=all bash scripts/tool/run_ctfm_frozen.sh
#   PROFILE=full_inspect TASK=prognosis COHORT=pe GPUS=0,1 bash scripts/tool/run_ctfm_frozen.sh
#   ACTION=preflight PROFILE=full_inspect TASK=diagnosis bash scripts/tool/run_ctfm_frozen.sh
#   WORKERS=12 NUM_WORKERS=12 bash scripts/tool/run_ctfm_frozen.sh    # upper bounds, capped by free RAM
#   FEATURE_INPUT=grid bash scripts/tool/run_ctfm_frozen.sh             # read full grids, not pooled copies
#
# WORKERS (prepare's preprocessing processes) and NUM_WORKERS (train/evaluate DataLoader
# workers) default to auto: sized from this machine's CPUs and free RAM, so one command fits a
# 4-vCPU/15 GB and a 16-vCPU/64 GB VM. A number is an upper bound; the log says when free
# memory lowered it. FEATURE_INPUT=pooled (default) trains on the pooled [513,1,1,1] copies
# kept in RAM; grid reads the 5.9 MB grids every epoch (same predictions, far slower).
#
# Every stage skips work that is already done, so re-running a command picks up where the last
# run stopped: prepare exits at once when the CT-FM cache is complete (VERIFY_CACHE=1 re-checks
# every study, REBUILD_CACHE=1 recomputes them), train is skipped when epoch_<EPOCHS> already
# holds a completed run with the same settings, and evaluate when that run already has its
# test result.csv. A completed run made with other settings (e.g. another EARLY_STOPPING) is
# an error rather than a silent skip. OVERWRITE=1 turns the skipping off and replaces the run.
#
# The base dataset manifests must already exist. This script never calls the dataset builder
# and never creates a train/validation/test split.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
PROFILE="${PROFILE:-full_inspect}"
TASK="${TASK:-diagnosis}"
COHORT="${COHORT:-all}"
TARGET="${TARGET:-}"
ACTION="${ACTION:-all}"
PYTHON="${PYTHON:-python3}"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOPPING="${EARLY_STOPPING:-15}"
NUM_WORKERS="${NUM_WORKERS:-auto}"
WORKERS="${WORKERS:-auto}"
FEATURE_INPUT="${FEATURE_INPUT:-pooled}"
case "$FEATURE_INPUT" in
  pooled) FEATURE_ARGS=() ;;
  grid) FEATURE_ARGS=(--set "data.file_column=image_path" --set "data.preload_inputs=false") ;;
  *) printf 'error: FEATURE_INPUT must be pooled or grid (got %s)\n' "$FEATURE_INPUT" >&2; exit 2 ;;
esac
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
    MANIFEST="manifests/ct_fm/diagnosis.csv"
    RUN_ID="DX_ctfm_frozen"
    ;;
  prognosis)
    case "$COHORT" in
      all|all_patient)
        CONFIG="${PROJECT_ROOT}/configs/runs/01_foundation/ct_fm_frozen_prognosis_all.yaml"
        MANIFEST="manifests/ct_fm/prognosis_all_patient.csv"
        RUN_ID="PR_ctfm_frozen_all"
        ;;
      pe|pe_positive|PE_positive)
        CONFIG="${PROJECT_ROOT}/configs/runs/01_foundation/ct_fm_frozen_prognosis_pe.yaml"
        MANIFEST="manifests/ct_fm/prognosis_pe_positive.csv"
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
  # CT-FM features (upstream contract: SPL, 3x1x1 mm, 24x128x128 patches) are computed once
  # per study into <derived>/cache/<profile>/ct_fm/ and manifests/ct_fm/*.csv. A finished
  # cache is recognised from its dataset.json in seconds; an unfinished one resumes, reusing
  # every study whose features match the current contract. VERIFY_CACHE=1 re-checks every
  # cached study (~5 studies/s over the bucket mount, over an hour on full_inspect).
  # REBUILD_CACHE=1 recomputes every study; OVERWRITE=1 only replaces
  # this EPOCHS run's epoch_<EPOCHS>/ folder (on full_inspect a cache rebuild is ~20 h, not a
  # run reset). Other EPOCHS values of the same run are separate folders and never collide.
  PREPARE_ARGS=(--device "$([[ -n "$GPUS" ]] && echo "cuda:${GPUS%%,*}" || echo cpu)" --workers "$WORKERS")
  [[ -n "${REBUILD_CACHE:-}" ]] && PREPARE_ARGS+=(--overwrite)
  [[ -n "${VERIFY_CACHE:-}" ]] && PREPARE_ARGS+=(--verify)
  "$PYTHON" "${PROJECT_ROOT}/tools/data/build_ctfm_cache.py" \
    --dataset-root "$DATASET_ROOT" \
    --raw-root "$RAW_ROOT" \
    --output-name ct_fm \
    --quiet \
    "${PREPARE_ARGS[@]}" 2>&1 | tee -a "${DATASET_ROOT}/logs.txt"
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
TRAIN_ARGS+=("${TARGET_ARGS[@]}" "${FEATURE_ARGS[@]}")
[[ -n "${BATCH_SIZE:-}" ]] && TRAIN_ARGS+=(--set "training.batch_size=${BATCH_SIZE}")
TRAIN_ARGS+=(--set "compute.num_workers=${NUM_WORKERS}")
[[ -n "${OVERWRITE:-}" ]] && TRAIN_ARGS+=(--overwrite)

# Manifests written before pooled features existed lack pooled_path; prepare adds it by
# pooling the cached grids (no CT-FM recomputation).
require_feature_column() {
  [[ "$FEATURE_INPUT" == "pooled" ]] || return 0
  head -1 "${DATASET_ROOT}/${MANIFEST}" | tr -d '"\r' | tr ',' '\n' | grep -qx 'pooled_path' || {
    printf 'error: %s has no pooled_path column\n' "${DATASET_ROOT}/${MANIFEST}" >&2
    printf '       run ACTION=prepare once (cached grids are reused; only the pooled copies are written)\n' >&2
    printf '       or set FEATURE_INPUT=grid to read the full grids\n' >&2
    exit 2; }
}

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
  PREFLIGHT_ARGS+=("${TARGET_ARGS[@]}" "${FEATURE_ARGS[@]}")
  "$PYTHON" tools/preflight.py "${PREFLIGHT_ARGS[@]}"
  exit $?
fi

if [[ "$ACTION" == "dry" ]]; then
  cd "$PROJECT_ROOT"
  "$PYTHON" tools/launch.py "${TRAIN_ARGS[@]}" --dry-run
  exit $?
fi

# epoch_<N>/logs.txt is written by the Python entry points themselves: training writes the
# run log and evaluation appends its section (thresholds, split counts, warnings, final test
# block) to the same file. Every logged line is also echoed to this terminal.

# The run's epoch_<EPOCHS>/ folder and how far it got (absent, incomplete, trained, evaluated,
# or different: completed with other settings), resolved by tools/run_status.py exactly as
# train_task.py and evaluate.py do.
RUN_STATE=absent
EPOCH_DIR=""
RUN_DIFFERENCE=""
if [[ "$ACTION" == "train" || "$ACTION" == "evaluate" || "$ACTION" == "all" ]]; then
  STATUS_LINE="$(cd "$PROJECT_ROOT" && "$PYTHON" tools/run_status.py "${TRAIN_ARGS[@]}")"
  read -r RUN_STATE EPOCH_DIR RUN_DIFFERENCE <<< "$STATUS_LINE"
  if [[ -z "${OVERWRITE:-}" && "$RUN_STATE" == "different" ]]; then
    printf 'error: %s holds a completed run made with other settings (differs in: %s)\n' \
      "$EPOCH_DIR" "$RUN_DIFFERENCE" >&2
    printf '       rerun with the same settings to reuse it, or OVERWRITE=1 to replace it\n' >&2
    exit 2
  fi
fi

if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
  if [[ ! -f "${DATASET_ROOT}/${MANIFEST}" ]]; then
    printf 'error: CT-FM manifest not found: %s\n' "${DATASET_ROOT}/${MANIFEST}" >&2
    exit 2
  fi
  require_feature_column
  cd "$PROJECT_ROOT"
  if [[ -z "${OVERWRITE:-}" && ( "$RUN_STATE" == "trained" || "$RUN_STATE" == "evaluated" ) ]]; then
    printf '==> train: skipped, %s already holds a completed run (OVERWRITE=1 retrains)\n' "$EPOCH_DIR"
  elif [[ -z "${OVERWRITE:-}" && "$RUN_STATE" == "incomplete" ]]; then
    # Task training cannot resume; say so here instead of failing inside the launcher.
    printf 'error: %s holds an unfinished run; OVERWRITE=1 replaces it\n' "$EPOCH_DIR" >&2
    exit 2
  else
    # Use the launcher so GPUS=0,1 becomes a real torchrun/DDP job rather than a
    # single process that merely records two device IDs in the config.
    "$PYTHON" tools/launch.py --quiet "${TRAIN_ARGS[@]}"
    RUN_STATE=trained
  fi
fi

if [[ "$ACTION" == "evaluate" || "$ACTION" == "all" ]]; then
  if [[ ! -f "${DATASET_ROOT}/${MANIFEST}" ]]; then
    printf 'error: CT-FM manifest not found: %s\n' "${DATASET_ROOT}/${MANIFEST}" >&2
    exit 2
  fi
  require_feature_column
  cd "$PROJECT_ROOT"
  if [[ -z "${OVERWRITE:-}" && "$RUN_STATE" == "evaluated" ]]; then
    printf '==> evaluate: skipped, %s/result.csv already exists (OVERWRITE=1 re-evaluates)\n' "$EPOCH_DIR"
  else
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
    EVAL_ARGS+=("${TARGET_ARGS[@]}" "${FEATURE_ARGS[@]}")
    EVAL_ARGS+=(--set "compute.num_workers=${NUM_WORKERS}")
    [[ -n "${OVERWRITE:-}" ]] && EVAL_ARGS+=(--overwrite)
    "$PYTHON" tools/launch.py --quiet "${EVAL_ARGS[@]}"
  fi
fi

if [[ "$ACTION" == "train" || "$ACTION" == "all" || "$ACTION" == "evaluate" ]]; then
  printf '\n==> completed task=%s cohort=%s profile=%s\n' "$TASK" "$COHORT" "$PROFILE"
  printf '    epochs=%s early_stopping_patience=%s gpus=%s\n' "$EPOCHS" "$EARLY_STOPPING" "$GPUS"
  # Each EPOCHS budget is its own folder: <run>/epoch_<EPOCHS>/ (config and result.json too).
  printf '    output=%s\n' "$EPOCH_DIR"
  if [[ -d "$EPOCH_DIR" ]]; then
    printf '    result.csv=%s/result.csv (metrics per split)\n' "$EPOCH_DIR"
    printf '    predictions.csv=%s/predictions.csv\n' "$EPOCH_DIR"
    printf '    logs.txt=%s/logs.txt\n' "$EPOCH_DIR"
    printf '    training_curves=%s/training_curves.png\n' "$EPOCH_DIR"
    printf '    preview=%s/preview/\n' "$EPOCH_DIR"
  fi
fi
