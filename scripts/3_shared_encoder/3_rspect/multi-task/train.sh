#!/usr/bin/env bash
# External RSPECT/RSNA-STR multitask supervised transfer.  WEIGHT_SOURCE selects
# pretrained, dapt, alignment or custom input weights; the shared runner owns mapping.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../_matrix_runner.sh"
RSPECT_MODE=multitask matrix_run_rspect "$@"
