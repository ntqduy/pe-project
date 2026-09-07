#!/usr/bin/env bash
# Global image plus PESI.
#
# Thin wrapper: resolves prog.image_pesi through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.image_pesi "$@"
