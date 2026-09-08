#!/usr/bin/env bash
# Pretrained baseline: frozen public TotalFM encoder, fixed PE probe head.
#
# Thin wrapper: resolves repr.foundation.totalfm through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.foundation.totalfm "$@"
