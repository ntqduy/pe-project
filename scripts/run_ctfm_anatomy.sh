#!/usr/bin/env bash
# CT-FM frozen proposal: train organ adapters + either concat or Soft-MoE fusion.
# Diagnosis is image-only PE classification. Prognosis is image-only multitask prediction
# over the seven INSPECT endpoints, with either all-patient or PE-positive official manifests.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/use_gcs_storage.sh"
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
    MANIFEST="manifests/ct_fm/diagnosis.csv"
    RUN_ID="DX_ctfm_anatomy_concat"
    ;;
  diagnosis:all:moe)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_diagnosis_moe.yaml"
    MANIFEST="manifests/ct_fm/diagnosis.csv"
    RUN_ID="DX_ctfm_anatomy_moe"
    ;;
  prognosis:all:concat)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_prognosis_concat.yaml"
    MANIFEST="manifests/ct_fm/prognosis_all_patient.csv"
    RUN_ID="PR_ctfm_anatomy_concat_all"
    ;;
  prognosis:all:moe)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_prognosis_moe.yaml"
    MANIFEST="manifests/ct_fm/prognosis_all_patient.csv"
    RUN_ID="PR_ctfm_anatomy_moe_all"
    ;;
  prognosis:pe:concat)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_prognosis_concat.yaml"
    MANIFEST="manifests/ct_fm/prognosis_pe_positive.csv"
    RUN_ID="PR_ctfm_anatomy_concat_pe"
    ;;
  prognosis:pe:moe)
    CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/ct_fm_frozen_anatomy_prognosis_moe.yaml"
    MANIFEST="manifests/ct_fm/prognosis_pe_positive.csv"
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
  # Resumable CT-FM feature cache (see scripts/run_ctfm_frozen.sh): studies whose contract
  # fingerprint still matches are reused. REBUILD_CACHE=1 recomputes every study; OVERWRITE=1
  # only replaces this EPOCHS run's epoch_<EPOCHS>/ folder.
  PREPARE=(--dataset-root "$DATASET_ROOT" --raw-root "$RAW_ROOT" --output-name ct_fm --quiet
    --device "$([[ -n "$GPUS" ]] && echo "cuda:${GPUS%%,*}" || echo cpu)")
  [[ -n "${REBUILD_CACHE:-}" ]] && PREPARE+=(--overwrite)
  "$PYTHON" "$PROJECT_ROOT/tools/data/build_ctfm_cache.py" "${PREPARE[@]}" 2>&1 | tee -a "$DATASET_ROOT/logs.txt"
fi

# The organ branches pool CT-FM features inside the ROI masks of this dataset profile. Stage
# run ids carry the profile except for full_inspect (SEG_pseudo_anatomy__ds_<profile>, see
# stamp_experiment_variant), while the configs name the full_inspect runs, so point them at
# the matching runs here. SEGMENTATION_RUN / ROI_RUN (paths under outputs/) override.
PROFILE_SUFFIX=""
[[ "$PROFILE" != "full_inspect" ]] && PROFILE_SUFFIX="__ds_${PROFILE}"
SEGMENTATION_RUN="${SEGMENTATION_RUN:-segmentation/SEG_pseudo_anatomy${PROFILE_SUFFIX}}"
ROI_RUN="${ROI_RUN:-roi/ROI_anatomy_and_controls${PROFILE_SUFFIX}}"
OUTPUT_ROOT="${PE_CLOUD_PROJECT_ROOT:-${PE_CLOUD_ROOT:-}/pe-project}/outputs"
if [[ "$ACTION" != "prepare" ]]; then
  [[ -d "$OUTPUT_ROOT/$SEGMENTATION_RUN" ]] || {
    echo "error: no segmentation run for PROFILE=$PROFILE at $OUTPUT_ROOT/$SEGMENTATION_RUN" >&2
    echo "       run: PROFILE=$PROFILE bash scripts/run_segmentation.sh  (or set SEGMENTATION_RUN=...)" >&2
    exit 2; }
  [[ -f "$OUTPUT_ROOT/$ROI_RUN/roi_manifest.csv" ]] || {
    echo "error: no ROI run for PROFILE=$PROFILE at $OUTPUT_ROOT/$ROI_RUN (roi_manifest.csv missing)" >&2
    echo "       run: PROFILE=$PROFILE bash scripts/run_roi.sh  (or set ROI_RUN=...)" >&2
    exit 2; }
fi

COMMON=(--config "$CONFIG" --gpus "$GPUS"
  --set "data.profile=$PROFILE"
  --set "data.manifest=$MANIFEST"
  --set "data.roi_manifest=$ROI_RUN/roi_manifest.csv"
  --set "supervision.masks=$SEGMENTATION_RUN"
  --set "supervision.rois=$ROI_RUN"
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

