#!/usr/bin/env bash
# Build the CT-FM feature cache once per dataset profile, before any ctfm_frozen_3d case
# (tools/data/build_ctfm_cache.py). It never creates a split: it reads the profile's
# official-split manifests and writes manifests/ct_fm/{diagnosis,prognosis_*}.csv whose
# pooled_path points at per-study pooled CT-FM features (CT-FM upstream contract: SPL,
# 3x1x1 mm, 24x128x128 patches). A finished cache is recognised in seconds; an unfinished one
# resumes, reusing every study that matches the current contract.
#
#   bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh                       # full_inspect, cuda:0
#   PROFILE=smoke_30 GPUS='' bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh   # CPU
#   VERIFY_CACHE=1 ...   re-check every cached study (~5 studies/s over the bucket mount)
#   REBUILD_CACHE=1 ...  recompute every study (~20 h on full_inspect)
#   WORKERS=auto         preprocessing processes, sized from free CPUs / RAM
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
# shellcheck source=../../tool/_flags.sh
source "$PROJECT_ROOT/scripts/tool/_flags.sh"
PROFILE="${PROFILE:-full_inspect}"
GPUS="${GPUS-0}"
WORKERS="${WORKERS:-auto}"
PYTHON="${PYTHON:-python3}"
DATASET_ROOT="${PE_DERIVED_ROOT}/datasets/${PROFILE}"

if [[ ! -f "${DATASET_ROOT}/manifests/diagnosis.csv" ]]; then
  printf 'error: official-split manifests not found under %s\n' "${DATASET_ROOT}" >&2
  printf '       build the dataset profile first (scripts/data/preprocessing.sh); this script never creates a split\n' >&2
  exit 2
fi

ARGS=(--device "$([[ -n "$GPUS" ]] && echo "cuda:${GPUS%%,*}" || echo cpu)" --workers "$WORKERS")
if is_true REBUILD_CACHE; then ARGS+=(--overwrite); fi
if is_true VERIFY_CACHE; then ARGS+=(--verify); fi
"$PYTHON" "${PROJECT_ROOT}/tools/data/build_ctfm_cache.py" \
  --dataset-root "$DATASET_ROOT" \
  --raw-root "$PE_RAW_INSPECT_ROOT" \
  --output-name ct_fm \
  --quiet \
  "${ARGS[@]}" 2>&1 | tee -a "${DATASET_ROOT}/logs.txt"
