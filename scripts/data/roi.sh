#!/usr/bin/env bash
# Build ROI1-ROI8 from a completed segmentation run. CPU workers are used for ROI creation.
# Existing patient split assignments are inherited from the source manifests and never moved.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
# shellcheck source=../tool/_flags.sh
source "$PROJECT_ROOT/scripts/tool/_flags.sh"
PROFILE="${PROFILE:-full_inspect}"
ACTION="${ACTION:-run}"
PYTHON="${PYTHON:-python3}"

case "$PROFILE" in smoke_30|test_500_sample|full_inspect) ;; *) echo "error: invalid PROFILE=$PROFILE" >&2; exit 2 ;; esac
case "$ACTION" in run|preflight) ;; *) echo "error: ACTION must be run or preflight" >&2; exit 2 ;; esac

# Segmentation output ids carry the dataset profile (SEG_pseudo_anatomy__ds_<profile>) except
# for full_inspect, the baseline profile that is never stamped, so the matching run is the
# default; SEGMENTATION_RUN=<path under outputs/> overrides it.
if [[ -z "${SEGMENTATION_RUN:-}" ]]; then
  SEGMENTATION_RUN="segmentation/SEG_pseudo_anatomy"
  [[ "$PROFILE" != "full_inspect" ]] && SEGMENTATION_RUN+="__ds_${PROFILE}"
  if [[ ! -d "${PE_CLOUD_PROJECT_ROOT}/outputs/${SEGMENTATION_RUN}" ]]; then
    printf 'error: no segmentation run for PROFILE=%s at %s\n' "$PROFILE" \
      "${PE_CLOUD_PROJECT_ROOT}/outputs/${SEGMENTATION_RUN}" >&2
    printf '       run: PROFILE=%s bash scripts/data/segmentation.sh  (or set SEGMENTATION_RUN=...)\n' "$PROFILE" >&2
    exit 2
  fi
fi

ARGS=(--set "data.profile=${PROFILE}" --set "data.manifest=manifests/ctpa.csv")
[[ -n "${ROI_WORKERS:-}" ]] && ARGS+=(--set "roi.workers=${ROI_WORKERS}")
ARGS+=(--set "roi.segmentation_run=${SEGMENTATION_RUN}")
if is_true OVERWRITE; then ARGS+=(--overwrite); fi

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

