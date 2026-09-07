#!/usr/bin/env bash
# Frozen-model heart removal (necessity). No retraining, no re-segmentation.
#
# Thin wrapper: resolves anatomy.remove_heart through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped anatomy.remove_heart "$@"
