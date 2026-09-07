#!/usr/bin/env bash
# Fixed frozen-encoder 30-day mortality probe.
#
# Thin wrapper: resolves probe.prog through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run probe.prog "$@"
