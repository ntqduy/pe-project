#!/usr/bin/env bash
# CT-FM frozen proposal: train organ adapters + either concat or Soft-MoE fusion.
# Diagnosis is image-only PE classification. Prognosis is image-only multitask prediction
# over the seven INSPECT endpoints, with either all-patient or PE-positive official manifests.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-full_inspect}"
TASK="${TASK:-diagnosis}"
COHORT="${COHORT:-all}"
FUSION="${FUSION:-concat}"
ACTION="${ACTION:-all}"
PYTHON="${PYTHON:-python3}"
GPUS="${GPUS:-0}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING="${EARLY_STOPPING:-10}"

case "$TASK:$COHORT:$FUSION" in
  diagnosis:all:concat)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_diagnosis_concat.yaml"
    MANIFEST="ct_fm_frozen/manifests/diagnosis.csv"
    RUN_ID="DX_ctfm_anatomy_concat"
    ;;
  diagnosis:all:moe)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_diagnosis_moe.yaml"
    MANIFEST="ct_fm_frozen/manifests/diagnosis.csv"
    RUN_ID="DX_ctfm_anatomy_moe"
    ;;
  prognosis:all:concat)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_prognosis_concat.yaml"
    MANIFEST="ct_fm_frozen/manifests/prognosis_all_patient.csv"
    RUN_ID="PR_ctfm_anatomy_concat_all"
    ;;
  prognosis:all:moe)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_prognosis_moe.yaml"
    MANIFEST="ct_fm_frozen/manifests/prognosis_all_patient.csv"
    RUN_ID="PR_ctfm_anatomy_moe_all"
    ;;
  prognosis:pe:concat)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_prognosis_concat.yaml"
    MANIFEST="ct_fm_frozen/manifests/prognosis_pe_positive.csv"
    RUN_ID="PR_ctfm_anatomy_concat_pe"
    ;;
  prognosis:pe:moe)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_prognosis_moe.yaml"
    MANIFEST="ct_fm_frozen/manifests/prognosis_pe_positive.csv"
    RUN_ID="PR_ctfm_anatomy_moe_pe"
    ;;
  *)
    echo "error: use TASK=diagnosis|prognosis, COHORT=all|pe, FUSION=concat|moe" >&2
    exit 2
    ;;
esac

case "$ACTION" in prepare|train|evaluate|all|preflight|dry) ;; *) echo "error: invalid ACTION=$ACTION" >&2; exit 2 ;; esac
case "$EPOCHS" in ''|*[!0-9]*) echo "error: EPOCHS must be positive" >&2; exit 2 ;; esac
case "$EARLY_STOPPING" in ''|*[!0-9]*) echo "error: EARLY_STOPPING must be positive" >&2; exit 2 ;; esac
(( EPOCHS >= 1 && EARLY_STOPPING >= 1 )) || { echo "error: EPOCHS/EARLY_STOPPING must be >= 1" >&2; exit 2; }

RAW_ROOT="${PE_RAW_INSPECT_ROOT:-/mnt/Stanford_INSPECT_dataset}"
if [[ -n "${PE_DERIVED_ROOT:-}" ]]; then
  DERIVED_ROOT="$PE_DERIVED_ROOT"
elif [[ -n "${PE_CLOUD_ROOT:-}" && -d "$PE_CLOUD_ROOT/derived" ]]; then
  DERIVED_ROOT="$PE_CLOUD_ROOT/derived"
else
  DERIVED_ROOT="${PE_CLOUD_ROOT:-}/data/derived"
fi
DATASET_ROOT="$DERIVED_ROOT/datasets/$PROFILE"

if [[ "$ACTION" == "prepare" || "$ACTION" == "all" ]]; then
  [[ -f "$DATASET_ROOT/manifests/diagnosis.csv" ]] || {
    echo "error: official dataset profile is missing; run scripts/run_preprocessing.sh first" >&2; exit 2; }
  if [[ ! -d "$DATASET_ROOT/ct_fm_frozen" || -n "${OVERWRITE:-}" ]]; then
    PREPARE=(--dataset-root "$DATASET_ROOT" --raw-root "$RAW_ROOT" --output-name ct_fm_frozen)
    [[ -n "${OVERWRITE:-}" ]] && PREPARE+=(--overwrite)
    "$PYTHON" "$PROJECT_ROOT/tools/data/build_ctfm_cache.py" "${PREPARE[@]}"
  fi
fi

COMMON=(--config "$CONFIG" --gpus "$GPUS"
  --set "data.profile=$PROFILE"
  --set "data.manifest=$MANIFEST"
  --set "experiment.id=$RUN_ID"
  --set "training.epochs=$EPOCHS"
  --set "training.early_stopping_patience=$EARLY_STOPPING")
[[ -n "${OVERWRITE:-}" ]] && COMMON+=(--overwrite)

cd "$PROJECT_ROOT"
if [[ "$ACTION" == "preflight" ]]; then
  exec "$PYTHON" tools/preflight.py "${COMMON[@]}"
fi
if [[ "$ACTION" == "dry" ]]; then
  exec "$PYTHON" tools/launch.py "${COMMON[@]}" --dry-run
fi
if [[ "$ACTION" == "train" || "$ACTION" == "all" ]]; then
  "$PYTHON" tools/launch.py "${COMMON[@]}"
fi
if [[ "$ACTION" == "evaluate" || "$ACTION" == "all" ]]; then
  "$PYTHON" tools/launch.py --evaluate "${COMMON[@]}" --allow-full
fi

echo "completed: task=$TASK cohort=$COHORT fusion=$FUSION epochs=$EPOCHS early_stopping=$EARLY_STOPPING"

