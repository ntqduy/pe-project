#!/usr/bin/env bash
# Image-report alignment. Produces C0.
#
# Thin wrapper: resolves repr.align through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.align "$@"
