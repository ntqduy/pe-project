#!/usr/bin/env bash
# Silver labels from: rule. The name is the cascade, in order.
#
# Thin wrapper: resolves data.silver.rule through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.silver.rule "$@"
