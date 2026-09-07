#!/usr/bin/env bash
# Frozen-model volume-matched random removal (necessity control).
#
# Thin wrapper: resolves anatomy.remove_random through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped anatomy.remove_random "$@"
