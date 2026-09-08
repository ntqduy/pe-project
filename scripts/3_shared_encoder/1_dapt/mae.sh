#!/usr/bin/env bash
# MAE domain-adaptive pretraining, starting from the original pretrained backbone.
#
# Thin wrapper: resolves repr.dapt.mae through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.dapt.mae "$@"
