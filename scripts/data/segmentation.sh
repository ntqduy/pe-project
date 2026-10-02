#!/usr/bin/env bash
# Generate and QC pseudo-anatomy masks from the existing official-split CTPA manifest.
# This stage never creates or changes train/validation/test assignments.
#
#   PROFILE=smoke_30 GPUS=0 bash scripts/data/segmentation.sh              # TotalSegmentator + LungMask
#   LUNGMASK=0 PROFILE=smoke_30 GPUS=0 bash scripts/data/segmentation.sh   # TotalSegmentator only
#
# Every mask comes from TotalSegmentator either way; LungMask only adds lung_lungmask and the
# lung cross-model Dice QC. Decide before a run starts: resuming it with the other setting is
# refused because the resolved config no longer matches.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
# shellcheck source=../tool/_flags.sh
source "$PROJECT_ROOT/scripts/tool/_flags.sh"
PROFILE="${PROFILE:-full_inspect}"
ACTION="${ACTION:-run}"
GPUS="${GPUS-0}"  # GPUS= (empty) means CPU; only an unset GPUS defaults to GPU 0
PYTHON="${PYTHON:-python3}"

case "$PROFILE" in smoke_30|test_500_sample|full_inspect) ;; *) echo "error: invalid PROFILE=$PROFILE" >&2; exit 2 ;; esac
case "$ACTION" in run|preflight) ;; *) echo "error: ACTION must be run or preflight" >&2; exit 2 ;; esac

ARGS=(--set "data.profile=${PROFILE}" --set "data.manifest=manifests/ctpa.csv")
if is_true OVERWRITE; then ARGS+=(--overwrite); fi
if is_false LUNGMASK; then ARGS+=(--set "segmentation.lungmask.enabled=false"); fi
[[ -n "${SEGMENTATION_DEVICES:-}" ]] && ARGS+=(--set "segmentation.devices=${SEGMENTATION_DEVICES}")

if [[ "$ACTION" == "preflight" ]]; then
  cd "$PROJECT_ROOT"
  exec "$PYTHON" run.py preflight data.segmentation --gpus "$GPUS" "${ARGS[@]}"
fi

SCOPE=()
if [[ -n "${PATIENT_ID:-}" ]]; then
  SCOPE=(--patient-id "$PATIENT_ID")
elif [[ -n "${MAX_CASES:-}" ]]; then
  SCOPE=(--max-cases "$MAX_CASES")
else
  SCOPE=(--allow-full)
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" run.py run data.segmentation --gpus "$GPUS" "${ARGS[@]}" "${SCOPE[@]}"

