#!/usr/bin/env bash
# Anatomy-aware multimodal prognosis, Soft-MoE fusion.
#
# Thin wrapper: resolves prog.anatomy.moe through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.anatomy.moe "$@"
