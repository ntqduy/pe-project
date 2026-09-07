#!/usr/bin/env bash
# SL01 - deterministic rules plus Falcon extraction.
#
# Thin wrapper: resolves data.silver.rules_falcon through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.silver.rules_falcon "$@"
