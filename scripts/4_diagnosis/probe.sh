#!/usr/bin/env bash
# Fixed frozen-encoder PE probe. The comparison instrument for encoder initializations.
#
# Thin wrapper: resolves probe.diag through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run probe.diag "$@"
