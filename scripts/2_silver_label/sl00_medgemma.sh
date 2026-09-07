#!/usr/bin/env bash
# SL00 - report-derived silver labels from MedGemma alone.
#
# Thin wrapper: resolves data.silver.medgemma through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.silver.medgemma "$@"
