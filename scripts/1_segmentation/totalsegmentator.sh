#!/usr/bin/env bash
# TotalSegmentator pseudo-anatomy masks, with LungMask contributing an independent lung Dice cross-check.
#
# Thin wrapper: resolves data.segmentation through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.segmentation "$@"
