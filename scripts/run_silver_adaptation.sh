#!/usr/bin/env bash
# Optional silver-supervised encoder adaptation after silver_labels.csv exists.
# This invokes the reviewed repr.silver.medgemma contract (C0/RSPECT initialization); it is
# deliberately separate from the CT-FM-frozen anatomy proposal, which uses public CT-FM weights.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/use_gcs_storage.sh"
PROFILE="${PROFILE:-full_inspect}"
ACTION="${ACTION:-run}"
GPUS="${GPUS:-0}"
PYTHON="${PYTHON:-python3}"
CONFIG="$PROJECT_ROOT/configs/runs/02_representation/silver_adaptation/medgemma.yaml"

case "$PROFILE" in smoke_30|test_500_sample|full_inspect) ;; *) echo "error: invalid PROFILE=$PROFILE" >&2; exit 2 ;; esac
case "$ACTION" in run|preflight|dry) ;; *) echo "error: ACTION must be run, preflight or dry" >&2; exit 2 ;; esac

ARGS=(--config "$CONFIG" --gpus "$GPUS"
  --set "data.profile=$PROFILE"
  --set "data.manifest=manifests/ctpa.csv")
[[ -n "${OVERWRITE:-}" ]] && ARGS+=(--overwrite)

cd "$PROJECT_ROOT"
if [[ "$ACTION" == "preflight" ]]; then
  exec "$PYTHON" tools/preflight.py "${ARGS[@]}"
fi
if [[ "$ACTION" == "dry" ]]; then
  exec "$PYTHON" tools/launch.py "${ARGS[@]}" --dry-run
fi
exec "$PYTHON" tools/launch.py "${ARGS[@]}"

