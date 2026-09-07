#!/usr/bin/env bash
# Image plus clinical plus PESI, Soft-MoE fusion.
#
# Thin wrapper: resolves prog.global.moe through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.global.moe "$@"
