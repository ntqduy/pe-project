#!/usr/bin/env bash
# Anatomy-aware DAPT (whole volume vs PA-only views), starting from the original pretrained backbone.
#
# Thin wrapper: resolves repr.dapt.anatomy through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.dapt.anatomy "$@"
