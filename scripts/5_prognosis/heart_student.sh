#!/usr/bin/env bash
# Heart-only ROI prognosis student (sufficiency).
#
# Thin wrapper: resolves prog.heart_student through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.heart_student "$@"
