#!/usr/bin/env bash
# Frozen-encoder linear probe on the pe_positive binary label.
#
# The fixed instrument for comparing WEIGHT_SOURCE: nothing but the encoder weights
# changes between runs, so the score is attributable to the representation alone.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../_matrix_runner.sh"
DIAGNOSIS_MODE=single_task LABEL=pe_positive STRATEGY=probe matrix_run_diagnosis "$@"
