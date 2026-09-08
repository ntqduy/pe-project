#!/usr/bin/env bash
# Pretrained baseline: frozen public CT-FM encoder, fixed PE probe head. The backbone is never updated.
#
# Thin wrapper: resolves repr.foundation.ct_fm through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../_lib.sh"
pe_run repr.foundation.ct_fm "$@"
