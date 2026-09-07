#!/usr/bin/env bash
# DAPT control - no adaptation. Loads the original pretrained backbone directly.
#
# Thin wrapper: resolves repr.dapt.none through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.dapt.none "$@"
