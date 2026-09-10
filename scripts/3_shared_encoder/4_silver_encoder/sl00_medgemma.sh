#!/usr/bin/env bash
# Silver encoder adaptation after the RSPECT stage, using accepted SL00 labels.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_matrix_runner.sh"
SILVER_ENCODER_LABEL=sl00_medgemma matrix_run_silver "$@"
