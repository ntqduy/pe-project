#!/usr/bin/env bash
# Clinical record plus PESI, no imaging.
#
# Thin wrapper: resolves prog.ehr_pesi through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.ehr_pesi "$@"
