#!/usr/bin/env bash
# External RSPECT/RSNA-STR PE-positive-vs-negative supervised transfer.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../_matrix_runner.sh"
RSPECT_MODE=single_pe matrix_run_rspect "$@"
