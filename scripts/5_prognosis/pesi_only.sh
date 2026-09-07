#!/usr/bin/env bash
# PESI/sPESI severity score only.
#
# Thin wrapper: resolves prog.pesi through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.pesi "$@"
