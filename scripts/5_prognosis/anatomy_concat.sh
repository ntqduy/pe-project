#!/usr/bin/env bash
# Anatomy-aware image plus clinical plus PESI, concat fusion.
#
# Thin wrapper: resolves prog.anatomy.concat through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.anatomy.concat "$@"
