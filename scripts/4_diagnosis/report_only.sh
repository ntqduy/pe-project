#!/usr/bin/env bash
# Text-only ceiling: never sees the volume.
#
# Thin wrapper: resolves diag.report_only through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run diag.report_only "$@"
