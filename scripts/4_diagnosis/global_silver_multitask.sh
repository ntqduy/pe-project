#!/usr/bin/env bash
# Full-CT global features, accepted-silver auxiliary heads.
#
# Thin wrapper: resolves diag.global.silver_multitask through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run diag.global.silver_multitask "$@"
