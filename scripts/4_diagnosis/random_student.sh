#!/usr/bin/env bash
# Volume-matched random-region student (sufficiency control).
#
# Thin wrapper: resolves anatomy.random_student through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run anatomy.random_student "$@"
