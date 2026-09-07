#!/usr/bin/env bash
# PA-only student with frozen-teacher distillation.
#
# Thin wrapper: resolves anatomy.pa_student_kd through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run anatomy.pa_student_kd "$@"
