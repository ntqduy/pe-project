#!/usr/bin/env bash
# Three-label native diagnosis multitask run.  WEIGHT_SOURCE is handled centrally.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_matrix_runner.sh"
DIAGNOSIS_MODE=multi_task STRATEGY="${STRATEGY:-global}" matrix_run_diagnosis "$@"
