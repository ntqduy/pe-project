#!/usr/bin/env bash
# exp01_baselines | 3D | Mamba-MAE ViM-S 3D (MAE weights, needs mamba_ssm fork): MLP head, 100% train
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp01_baselines mamba_mae_3d "$@"
