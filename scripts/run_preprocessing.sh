#!/usr/bin/env bash
# Stage 0 end to end: build one dataset profile from the read-only INSPECT release.
#
#   bash scripts/run_preprocessing.sh                  # full_inspect, the whole cohort
#   PROFILE=smoke_30 bash scripts/run_preprocessing.sh # 30-patient technical rehearsal
#   PROFILE=test_500_sample bash scripts/run_preprocessing.sh
#   ACTION=preflight bash scripts/run_preprocessing.sh # check paths and contracts only
#   OVERWRITE=1 bash scripts/run_preprocessing.sh      # rebuild over an existing profile
#
# Writes <derived>/datasets/<profile>/: manifests/*.csv, the CT cache volumes/*.npy with
# geometry and patch sidecars, clinical/* (EHR + sPESI), data_quality.{md,json}, audit/*
# and dataset.json. It only ever reads the raw release.
#
# Runtime for full_inspect is hours, not minutes: every eligible study is header-checked,
# then resampled, cropped and cached. Rehearse on PROFILE=smoke_30 first.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/use_gcs_storage.sh"
PROFILE="${PROFILE:-full_inspect}"
ACTION="${ACTION:-run}"
PYTHON="${PYTHON:-python3}"

case "$PROFILE" in
  smoke_30|test_500_sample|full_inspect) ;;
  *) printf 'error: PROFILE must be smoke_30, test_500_sample or full_inspect (got %s)\n' "$PROFILE" >&2; exit 2 ;;
esac
case "$ACTION" in
  run|dry|preflight|plan|show) ;;
  *) printf 'error: ACTION must be run, dry, preflight, plan or show (got %s)\n' "$ACTION" >&2; exit 2 ;;
esac

# The storage bucket must already be mounted; the helper only supplies default paths.
RAW_ROOT="${PE_RAW_INSPECT_ROOT:-$PE_CLOUD_ROOT/data/Stanford_INSPECT_dataset}"
if [[ ! -d "$RAW_ROOT/CT/full" ]]; then
  printf 'error: raw release not found at %s/CT/full\n' "$RAW_ROOT" >&2
  printf '       set PE_RAW_INSPECT_ROOT to the directory that contains CT/full\n' >&2
  exit 2
fi

DERIVED_ROOT="${PE_DERIVED_ROOT:-$PE_CLOUD_ROOT/data/derived}"
DATASET_ROOT="$DERIVED_ROOT/datasets/$PROFILE"

# Dataset generation is intentionally idempotent at the wrapper level. The lower-level
# builder protects its destination by raising on any existing output, which is correct for
# a library call but noisy when a user reruns this script to continue the pipeline. Reuse a
# complete profile; only an explicit OVERWRITE=1 may rebuild it.
dataset_complete() {
  local required
  for required in \
    dataset.json \
    data_quality.json \
    audit/split_audit.json \
    manifests/ctpa.csv \
    manifests/diagnosis.csv \
    manifests/prognosis.csv \
    manifests/prognosis_all_patient.csv \
    manifests/prognosis_pe_positive.csv; do
    [[ -s "$DATASET_ROOT/$required" ]] || return 1
  done
  return 0
}

dataset_scope_matches() {
  "$PYTHON" - "$DATASET_ROOT/dataset.json" "${MAX_CASES:-}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
requested = None if not sys.argv[2] else int(sys.argv[2])
stored = (payload.get("scope") or {}).get("max_cases")
if stored is not None:
    stored = int(stored)
print("yes" if stored == requested else "no")
PY
}

if [[ "$ACTION" == "run" || "$ACTION" == "preflight" ]]; then
  if [[ -z "${OVERWRITE:-}" ]] && dataset_complete && [[ "$(dataset_scope_matches)" == "yes" ]]; then
    printf '==> skip preprocessing: complete profile already exists\n'
    printf '    profile  %s\n' "$PROFILE"
    printf '    dataset  %s\n' "$DATASET_ROOT"
    printf '    use OVERWRITE=1 to rebuild this profile\n'
    exit 0
  fi

  if [[ -d "$DATASET_ROOT" && -n "$(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" && -z "${OVERWRITE:-}" ]]; then
    printf 'error: dataset profile exists but is incomplete or was built with a different scope: %s\n' "$DATASET_ROOT" >&2
    printf '       inspect/remove only the incomplete profile, or use OVERWRITE=1 to rebuild it\n' >&2
    exit 2
  fi
fi

declare -a cmd=("$PYTHON" "$PROJECT_ROOT/run.py" "$ACTION" "data.dataset.$PROFILE")
# A generation stage needs an explicit scope; building a whole profile is never implicit.
[[ "$ACTION" == "run" || "$ACTION" == "dry" ]] && cmd+=(--allow-full)
[[ -n "${MAX_CASES:-}" ]] && cmd=("${cmd[@]/--allow-full/}") && cmd+=(--max-cases "$MAX_CASES")
[[ -n "${OVERWRITE:-}" ]] && cmd+=(--overwrite)
[[ -n "${GPUS:-}" ]] && cmd+=(--gpus "$GPUS")

printf '==> stage 0  [profile=%s action=%s]\n' "$PROFILE" "$ACTION"
printf '    raw      %s\n' "$RAW_ROOT"
printf '    derived  %s\n' "${PE_DERIVED_ROOT:-$PE_CLOUD_ROOT/data/derived}/datasets/$PROFILE"
exec "${cmd[@]}"
