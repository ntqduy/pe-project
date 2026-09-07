#!/usr/bin/env bash
# Full-CT global features, single PE head.
#
# Thin wrapper: resolves diag.global.single through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run diag.global.single "$@"
