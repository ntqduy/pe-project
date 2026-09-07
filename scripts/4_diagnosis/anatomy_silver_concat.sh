#!/usr/bin/env bash
# Anatomy branches with accepted-silver organ heads, concat fusion.
#
# Thin wrapper: resolves diag.anatomy.silver.concat through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run diag.anatomy.silver.concat "$@"
