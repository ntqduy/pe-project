#!/usr/bin/env bash
# Raw clinical values plus missingness indicators only.
#
# Thin wrapper: resolves prog.ehr through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.ehr "$@"
