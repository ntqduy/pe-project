#!/usr/bin/env bash
# Shared target for every prognosis strategy launcher.  The invoked symlink path encodes
# COHORT/EHR_PROFILE/TASK/STRATEGY, so the individual files contain no copied logic.
set -euo pipefail

INVOKED="${BASH_SOURCE[0]}"
if [[ "$INVOKED" != /* ]]; then
  INVOKED="$PWD/$INVOKED"
fi
TASK_DIR="$(cd "$(dirname "$INVOKED")" && pwd)"
TASK="$(basename "$TASK_DIR")"
EHR_PROFILE="$(basename "$(dirname "$TASK_DIR")")"
COHORT="$(basename "$(dirname "$(dirname "$TASK_DIR")")")"
STRATEGY="$(basename "$INVOKED" .sh)"

REAL_SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
source "$(dirname "$REAL_SCRIPT")/../_matrix_runner.sh"
matrix_run_prognosis "$@"
