#!/usr/bin/env bash
# Lung-parenchyma-only ROI prognosis student (sufficiency).
#
# Thin wrapper: resolves prog.lung_student through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.lung_student "$@"
