#!/usr/bin/env bash
# Source this file before running the PE pipeline to keep derived data and experiment
# outputs on the `gs://pe-study/pe-storage` prefix mounted at /mnt/pe-storage.
#
# Usage:
#   source scripts/use_gcs_storage.sh
#
# Source code remains local at /mnt/pe-project.  The raw INSPECT release remains
# read-only at /mnt/Stanford_INSPECT_dataset.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'source this file instead: source scripts/use_gcs_storage.sh\n' >&2
  exit 2
fi

PE_STORAGE_MOUNT="/mnt/pe-storage"
if ! mountpoint -q "$PE_STORAGE_MOUNT"; then
  printf 'error: %s is not mounted. Mount gs://pe-study/pe-storage there first.\n' "$PE_STORAGE_MOUNT" >&2
  return 2
fi

# `derived/` and `pe-project/outputs/` are created by the pipeline when first needed.
# Do not create them here: sourcing this helper must not write to the bucket.

export PE_CLOUD_ROOT="$PE_STORAGE_MOUNT"
export PE_RAW_INSPECT_ROOT="/mnt/Stanford_INSPECT_dataset"
export PE_DERIVED_ROOT="$PE_STORAGE_MOUNT/derived"
export PE_LOCAL_CACHE_ROOT="/mnt/pe-project/cache"
# This workspace provides python3 rather than a `python` alias; callers may still override it.
export PYTHON="${PYTHON:-python3}"

printf 'PE storage configured:\n'
printf '  raw (read-only): %s\n' "$PE_RAW_INSPECT_ROOT"
printf '  derived data:    %s\n' "$PE_DERIVED_ROOT"
printf '  outputs:         %s/pe-project/outputs\n' "$PE_CLOUD_ROOT"
printf '  local cache:     %s\n' "$PE_LOCAL_CACHE_ROOT"
printf '  Python:          %s\n' "$PYTHON"
