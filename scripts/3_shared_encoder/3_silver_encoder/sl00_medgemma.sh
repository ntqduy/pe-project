#!/usr/bin/env bash
# Adapt C0 with accepted SL00 silver labels.
#
# Thin wrapper: resolves repr.silver.medgemma through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.silver.medgemma "$@"
