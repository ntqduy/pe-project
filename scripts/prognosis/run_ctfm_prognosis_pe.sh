#!/usr/bin/env bash
# CT-FM frozen + trainable MLP for prognosis restricted to PE-positive patients.
# Filtering is performed inside the existing PE-positive manifest; splits are not reassigned.
#
# Examples:
#   PROFILE=smoke_30 EPOCHS=1 bash scripts/prognosis/run_ctfm_prognosis_pe.sh
#   PROFILE=full_inspect GPUS=0,1 EPOCHS=100 EARLY_STOPPING=15 bash scripts/prognosis/run_ctfm_prognosis_pe.sh
#   ACTION=preflight PROFILE=full_inspect bash scripts/prognosis/run_ctfm_prognosis_pe.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
TASK=prognosis COHORT=pe exec "${PROJECT_ROOT}/scripts/tool/run_ctfm_frozen.sh" "$@"
