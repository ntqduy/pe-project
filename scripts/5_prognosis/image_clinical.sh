#!/usr/bin/env bash
# Global image plus the raw clinical record.
#
# Thin wrapper: resolves prog.image_ehr through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.image_ehr "$@"
