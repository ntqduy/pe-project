#!/usr/bin/env bash
# Generate and QC pseudo-anatomy masks from the existing official-split CTPA manifest.
# This stage never creates or changes train/validation/test assignments.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-full_inspect}"
ACTION="${ACTION:-run}"
GPUS="${GPUS:-0}"
PYTHON="${PYTHON:-python3}"

case "$PROFILE" in smoke_30|test_500_sample|full_inspect) ;; *) echo "error: invalid PROFILE=$PROFILE" >&2; exit 2 ;; esac
case "$ACTION" in run|preflight) ;; *) echo "error: ACTION must be run or preflight" >&2; exit 2 ;; esac

ARGS=(--set "data.profile=${PROFILE}" --set "data.manifest=manifests/ctpa.csv")
[[ -n "${OVERWRITE:-}" ]] && ARGS+=(--overwrite)
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

