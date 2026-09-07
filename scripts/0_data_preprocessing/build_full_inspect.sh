#!/usr/bin/env bash
# Build the whole eligible INSPECT cohort. Same code path and preprocessing as the 500-patient profile.
#
# Thin wrapper: resolves data.dataset.full_inspect through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.dataset.full_inspect "$@"
