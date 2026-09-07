#!/usr/bin/env bash
# Image plus clinical plus PESI, concat fusion.
#
# Thin wrapper: resolves prog.image_ehr_pesi through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.image_ehr_pesi "$@"
