#!/usr/bin/env bash
# Generate accepted/abstained silver labels from the report table.
# Silver labels are auxiliary supervision only; they are never used as test gold labels.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
PROFILE="${PROFILE:-full_inspect}"
ACTION="${ACTION:-run}"
GPUS="${GPUS:-0}"
PYTHON="${PYTHON:-python3}"

case "$PROFILE" in smoke_30|test_500_sample|full_inspect) ;; *) echo "error: invalid PROFILE=$PROFILE" >&2; exit 2 ;; esac
case "$ACTION" in run|preflight) ;; *) echo "error: ACTION must be run or preflight" >&2; exit 2 ;; esac

ARGS=(--set "data.profile=${PROFILE}")
[[ -n "${OVERWRITE:-}" ]] && ARGS+=(--overwrite)

if [[ "$ACTION" == "preflight" ]]; then
  cd "$PROJECT_ROOT"
  exec "$PYTHON" run.py preflight data.silver.medgemma --gpus "$GPUS" "${ARGS[@]}"
fi

SCOPE=()
if [[ -n "${PATIENT_ID:-}" ]]; then
  SCOPE=(--patient-id "$PATIENT_ID")
elif [[ -n "${MAX_REPORTS:-}" ]]; then
  SCOPE=(--max-reports "$MAX_REPORTS")
else
  SCOPE=(--allow-full)
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" run.py run data.silver.medgemma --gpus "$GPUS" "${ARGS[@]}" "${SCOPE[@]}"

