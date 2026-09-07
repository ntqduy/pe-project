#!/usr/bin/env bash
# Adapt C0 with accepted SL02 silver labels. Produces C_silver.
#
# Thin wrapper: resolves repr.silver.hybrid through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.silver.hybrid "$@"
