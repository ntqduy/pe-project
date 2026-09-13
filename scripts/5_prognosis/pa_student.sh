#!/usr/bin/env bash
# Pulmonary-artery-only ROI prognosis student (sufficiency).
#
# Thin wrapper: resolves prog.pa_student through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.pa_student "$@"
