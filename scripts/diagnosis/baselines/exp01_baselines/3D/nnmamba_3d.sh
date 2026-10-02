#!/usr/bin/env bash
# exp01_baselines | 3D | nnMamba4cls 3D (scratch, needs mamba_ssm): MLP head, 100% train
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp01_baselines nnmamba_3d "$@"
