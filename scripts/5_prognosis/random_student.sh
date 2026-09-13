#!/usr/bin/env bash
# Volume-matched random-region prognosis student (sufficiency control).
#
# Thin wrapper: resolves prog.random_student through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.random_student "$@"
