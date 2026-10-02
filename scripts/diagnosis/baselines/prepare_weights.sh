#!/usr/bin/env bash
# Download every baseline's pretrained weights once (before parallel runs) and print the
# weight-status table (LOADED / SCRATCH / UNAVAILABLE per arm), also written to
# <outputs>/diagnosis/BASE/weights_status.md.
#   bash scripts/diagnosis/baselines/prepare_weights.sh
#   bash scripts/diagnosis/baselines/prepare_weights.sh --models vit_3d swin_3d
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
cd "$PROJECT_ROOT"
exec "${PYTHON:-python3}" tools/baselines/prepare_weights.py "$@"
