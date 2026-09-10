#!/usr/bin/env bash
# Silver encoder adaptation after the RSPECT stage, using accepted SL01 labels.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_matrix_runner.sh"
SILVER_ENCODER_LABEL=sl01_rules_falcon matrix_run_silver "$@"
