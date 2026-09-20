#!/usr/bin/env bash
# CT-FM frozen + trainable MLP for prognosis on all eligible patients.
# The official train/validation/test assignment is inherited from the manifest.
#
# Examples:
#   PROFILE=smoke_30 EPOCHS=1 bash scripts/run_ctfm_prognosis_all.sh
#   PROFILE=full_inspect GPUS=0,1 EPOCHS=50 EARLY_STOPPING=10 bash scripts/run_ctfm_prognosis_all.sh
#   ACTION=evaluate PROFILE=full_inspect bash scripts/run_ctfm_prognosis_all.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK=prognosis COHORT=all exec "${PROJECT_ROOT}/scripts/run_ctfm_frozen.sh" "$@"
