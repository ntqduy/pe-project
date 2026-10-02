#!/usr/bin/env bash
# Stage 0 end to end: build one dataset profile from the read-only INSPECT release.
#
#   bash scripts/data/preprocessing.sh                  # full_inspect, the whole cohort
#   PROFILE=smoke_30 bash scripts/data/preprocessing.sh # 30-patient technical rehearsal
#   PROFILE=test_500_sample bash scripts/data/preprocessing.sh
#   ACTION=preflight bash scripts/data/preprocessing.sh # check paths and contracts only
#   OVERWRITE=1 bash scripts/data/preprocessing.sh      # rebuild over an existing profile
#
# Writes manifests/*.csv, data_quality.md, dataset.json and logs.txt in the dataset.
# CT and clinical caches live under <derived>/cache/<profile>/.
#
# Runtime for full_inspect is hours, not minutes: every eligible study is header-checked,
# then resampled, cropped and cached. Rehearse on PROFILE=smoke_30 first.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
# shellcheck source=../tool/_flags.sh
source "$PROJECT_ROOT/scripts/tool/_flags.sh"
PROFILE="${PROFILE:-full_inspect}"
ACTION="${ACTION:-run}"
PYTHON="${PYTHON:-python3}"

case "$PROFILE" in
  smoke_30|test_500_sample|full_inspect) ;;
  *) printf 'error: PROFILE must be smoke_30, test_500_sample or full_inspect (got %s)\n' "$PROFILE" >&2; exit 2 ;;
esac
# No ACTION=dry: run.py refuses a dry run for every data stage (they write files); use
# ACTION=plan or show to see what would run.
case "$ACTION" in
  run|preflight|plan|show) ;;
  dry) printf 'error: ACTION=dry is not supported for data stages (run.py refuses it); use plan or show\n' >&2; exit 2 ;;
  *) printf 'error: ACTION must be run, preflight, plan or show (got %s)\n' "$ACTION" >&2; exit 2 ;;
esac
OVERWRITE_ON=0; if is_true OVERWRITE; then OVERWRITE_ON=1; fi

if [[ "$ACTION" == "run" || "$ACTION" == "preflight" ]]; then
  [[ -d "$PE_CLOUD_ROOT" ]] || {
    printf 'error: output root %s does not exist\n' "$PE_CLOUD_ROOT" >&2
    exit 2
  }
fi

# The raw release must already be mounted (gcsfuse); use_gcs_storage.sh supplies the paths.
# show / plan only read the registry, so only run / preflight need the release.
RAW_ROOT="$PE_RAW_INSPECT_ROOT"
if [[ ( "$ACTION" == "run" || "$ACTION" == "preflight" ) && ! -d "$RAW_ROOT/CT/full" ]]; then
  printf 'error: raw release not found at %s/CT/full\n' "$RAW_ROOT" >&2
  printf '       set PE_RAW_INSPECT_ROOT to the directory that contains CT/full\n' >&2
  exit 2
fi

DERIVED_ROOT="$PE_DERIVED_ROOT"
DATASET_ROOT="$DERIVED_ROOT/datasets/$PROFILE"

# Dataset generation is intentionally idempotent at the wrapper level. The lower-level
# builder protects its destination by raising on any existing output, which is correct for
# a library call but noisy when a user reruns this script to continue the pipeline. Reuse a
# complete profile; only an explicit OVERWRITE=1 may rebuild it.
dataset_complete() {
  local required
  for required in \
    dataset.json \
    data_quality.md \
    manifests/exclusions.csv \
    manifests/ctpa.csv \
    manifests/diagnosis.csv \
    manifests/prognosis.csv \
    manifests/prognosis_all_patient.csv \
    manifests/prognosis_pe_positive.csv; do
    [[ -s "$DATASET_ROOT/$required" ]] || return 1
  done
  [[ -d "$DERIVED_ROOT/cache/$PROFILE/volumes" ]] || return 1
  [[ -f "$DERIVED_ROOT/cache/$PROFILE/clinical/ehr_profiles.json" ]] || return 1
  [[ -f "$DERIVED_ROOT/cache/$PROFILE/clinical/spesi_status.json" ]] || return 1
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
  if [[ "$OVERWRITE_ON" != "1" ]] && dataset_complete && [[ "$(dataset_scope_matches)" == "yes" ]]; then
    printf '==> skip preprocessing: complete profile already exists\n'
    printf '    profile  %s\n' "$PROFILE"
    printf '    dataset  %s\n' "$DATASET_ROOT"
    printf '    use OVERWRITE=1 to rebuild this profile\n'
    exit 0
  fi

  if [[ -d "$DATASET_ROOT" && -n "$(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" && "$OVERWRITE_ON" != "1" ]]; then
    printf 'error: dataset profile exists but is incomplete or was built with a different scope: %s\n' "$DATASET_ROOT" >&2
    if [[ -d "$DATASET_ROOT/volumes" || -d "$DATASET_ROOT/clinical" || -d "$DATASET_ROOT/ct_fm_frozen" ]]; then
      printf '       this is the pre-refactor layout; rebuild the profile with OVERWRITE=1\n' >&2
    fi
    printf '       inspect/remove only the incomplete profile, or use OVERWRITE=1 to rebuild it\n' >&2
    exit 2
  fi
fi

declare -a cmd=("$PYTHON" "$PROJECT_ROOT/run.py" "$ACTION" "data.dataset.$PROFILE")
# `run.py show` and `run.py plan` take only the experiment name; the passthrough flags exist
# for run and preflight only, and argparse rejects them anywhere else.
if [[ "$ACTION" == "run" || "$ACTION" == "preflight" ]]; then
  # A generation stage needs an explicit scope; building a whole profile is never implicit.
  # preflight validates the whole configured input and takes no scope flag.
  if [[ "$ACTION" != "preflight" ]]; then
    if [[ -n "${MAX_CASES:-}" ]]; then
      cmd+=(--max-cases "$MAX_CASES")
    else
      cmd+=(--allow-full)
    fi
  fi
  if [[ "$OVERWRITE_ON" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  if [[ -n "${GPUS:-}" ]]; then
    cmd+=(--gpus "$GPUS")
  fi
fi

if [[ "$ACTION" == "run" ]]; then
  mkdir -p "$DATASET_ROOT"
  {
    printf '\n==> stage 0  [profile=%s action=%s time=%s]\n' "$PROFILE" "$ACTION" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '    raw      %s\n' "$RAW_ROOT"
    printf '    derived  %s\n' "$DATASET_ROOT"
    "${cmd[@]}"
  } 2>&1 | tee -a "$DATASET_ROOT/logs.txt"
else
  exec "${cmd[@]}"
fi
