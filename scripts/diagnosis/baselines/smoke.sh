#!/usr/bin/env bash
# One bf16 forward+backward per baseline arm at 128^3 on a GPU (real weights), with
# gradient, Grad-CAM-gradient and peak-VRAM checks. Run once on the VM before long runs.
#   bash scripts/diagnosis/baselines/smoke.sh
#   bash scripts/diagnosis/baselines/smoke.sh --models vmamba_3d mamba_mae_3d nnmamba_3d
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
cd "$PROJECT_ROOT"
exec "${PYTHON:-python3}" tools/baselines/smoke.py "$@"
