#!/usr/bin/env bash
# Anatomy-aware PE diagnosis (heart/PA/lung branches), the reference model.
#
# Thin wrapper: resolves diag.anatomy.concat through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run diag.anatomy.concat "$@"
