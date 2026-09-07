#!/usr/bin/env bash
# Frozen-model lung-parenchyma removal (necessity).
#
# Thin wrapper: resolves anatomy.remove_lung through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped anatomy.remove_lung "$@"
