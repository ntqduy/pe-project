#!/usr/bin/env bash
# Build ROI1-ROI8 from a completed segmentation run. CPU workers are used for ROI creation.
# Existing patient split assignments are inherited from the source manifests and never moved.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-full_inspect}"
ACTION="${ACTION:-run}"
PYTHON="${PYTHON:-python3}"

case "$PROFILE" in smoke_30|test_500_sample|full_inspect) ;; *) echo "error: invalid PROFILE=$PROFILE" >&2; exit 2 ;; esac
case "$ACTION" in run|preflight) ;; *) echo "error: ACTION must be run or preflight" >&2; exit 2 ;; esac

ARGS=(--set "data.profile=${PROFILE}" --set "data.manifest=manifests/ctpa.csv")
[[ -n "${ROI_WORKERS:-}" ]] && ARGS+=(--set "roi.workers=${ROI_WORKERS}")
[[ -n "${SEGMENTATION_RUN:-}" ]] && ARGS+=(--set "roi.segmentation_run=${SEGMENTATION_RUN}")
[[ -n "${OVERWRITE:-}" ]] && ARGS+=(--overwrite)

if [[ "$ACTION" == "preflight" ]]; then
  cd "$PROJECT_ROOT"
  exec "$PYTHON" run.py preflight data.roi "${ARGS[@]}"
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
exec "$PYTHON" run.py run data.roi "${ARGS[@]}" "${SCOPE[@]}"

