#!/usr/bin/env bash
# ROI1-ROI8 crops and volume-matched random controls, read from a finished segmentation run.
#
# Thin wrapper: resolves data.roi through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.roi "$@"
