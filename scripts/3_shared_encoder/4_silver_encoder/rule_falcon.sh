#!/usr/bin/env bash
# Silver encoder adaptation after the RSPECT stage, using accepted rule_falcon labels.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_matrix_runner.sh"
SILVER_ENCODER_LABEL=rule_falcon matrix_run_silver "$@"
