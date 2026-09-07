#!/usr/bin/env bash
# Frozen-model pulmonary-artery removal (necessity).
#
# Thin wrapper: resolves anatomy.remove_pa through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped anatomy.remove_pa "$@"
