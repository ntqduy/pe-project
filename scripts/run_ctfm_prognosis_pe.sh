#!/usr/bin/env bash
# CT-FM frozen + trainable MLP for prognosis restricted to PE-positive patients.
# Filtering is performed inside the existing PE-positive manifest; splits are not reassigned.
#
# Examples:
#   PROFILE=smoke_30 EPOCHS=1 bash scripts/run_ctfm_prognosis_pe.sh
#   PROFILE=full_inspect GPUS=0,1 EPOCHS=50 EARLY_STOPPING=10 bash scripts/run_ctfm_prognosis_pe.sh
#   ACTION=preflight PROFILE=full_inspect bash scripts/run_ctfm_prognosis_pe.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/use_gcs_storage.sh"
TASK=prognosis COHORT=pe exec "${PROJECT_ROOT}/scripts/run_ctfm_frozen.sh" "$@"
