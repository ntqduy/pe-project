#!/usr/bin/env bash
# DINO domain-adaptive pretraining, starting from the original pretrained backbone.
#
# Thin wrapper: resolves repr.dapt.dino through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.dapt.dino "$@"
