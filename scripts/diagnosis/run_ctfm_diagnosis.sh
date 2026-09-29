#!/usr/bin/env bash
# CT-FM frozen + trainable MLP for PE diagnosis.
#
# Examples:
#   PROFILE=smoke_30 EPOCHS=1 bash scripts/diagnosis/run_ctfm_diagnosis.sh
#   PROFILE=full_inspect GPUS=0,1 EPOCHS=100 EARLY_STOPPING=15 bash scripts/diagnosis/run_ctfm_diagnosis.sh
#   ACTION=preflight PROFILE=full_inspect bash scripts/diagnosis/run_ctfm_diagnosis.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
TASK=diagnosis COHORT=all exec "${PROJECT_ROOT}/scripts/tool/run_ctfm_frozen.sh" "$@"
