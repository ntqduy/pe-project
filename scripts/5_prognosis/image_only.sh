#!/usr/bin/env bash
# Global full-CT image only.
#
# Thin wrapper: resolves prog.image through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run prog.image "$@"
