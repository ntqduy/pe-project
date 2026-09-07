#!/usr/bin/env bash
# Build the 500-patient rehearsal cohort from the read-only INSPECT release.
#
# Thin wrapper: resolves data.dataset.test_500_sample through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.dataset.test_500_sample "$@"
