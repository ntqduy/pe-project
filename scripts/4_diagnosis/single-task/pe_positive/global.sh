#!/usr/bin/env bash
# PE-positive-vs-negative diagnosis run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../_matrix_runner.sh"
DIAGNOSIS_MODE=single_task LABEL=pe_positive STRATEGY="${STRATEGY:-global}" matrix_run_diagnosis "$@"
