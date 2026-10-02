#!/usr/bin/env bash
# exp01_baselines | 3D | PENet 3D (released PE weights): MLP head, 100% train
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp01_baselines penet_3d "$@"
