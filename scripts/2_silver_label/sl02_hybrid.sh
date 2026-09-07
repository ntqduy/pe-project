#!/usr/bin/env bash
# SL02 - rules plus Falcon plus MedGemma with agreement adjudication.
#
# Thin wrapper: resolves data.silver.hybrid through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.silver.hybrid "$@"
