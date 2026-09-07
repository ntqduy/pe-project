#!/usr/bin/env bash
# Lung-only ROI student (sufficiency).
#
# Thin wrapper: resolves anatomy.lung_student through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run anatomy.lung_student "$@"
