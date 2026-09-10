#!/usr/bin/env bash
# Acute-PE diagnosis run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../_matrix_runner.sh"
DIAGNOSIS_MODE=single_task LABEL=pe_acute STRATEGY="${STRATEGY:-global}" matrix_run_diagnosis "$@"
