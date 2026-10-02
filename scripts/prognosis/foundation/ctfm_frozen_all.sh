#!/usr/bin/env bash
# CT-FM frozen + trainable MLP for prognosis on all eligible patients.
# The official train/validation/test assignment is inherited from the manifest.
#
# Examples:
#   PROFILE=smoke_30 EPOCHS=1 bash scripts/prognosis/foundation/ctfm_frozen_all.sh
#   PROFILE=full_inspect GPUS=0,1 EPOCHS=100 EARLY_STOPPING=15 bash scripts/prognosis/foundation/ctfm_frozen_all.sh
#   ACTION=evaluate PROFILE=full_inspect bash scripts/prognosis/foundation/ctfm_frozen_all.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
TASK=prognosis COHORT=all exec "${PROJECT_ROOT}/scripts/tool/run_ctfm_frozen.sh" "$@"
