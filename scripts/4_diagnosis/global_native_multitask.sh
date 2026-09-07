#!/usr/bin/env bash
# Full-CT global features, native multitask heads.
#
# Thin wrapper: resolves diag.global.native_multitask through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run diag.global.native_multitask "$@"
