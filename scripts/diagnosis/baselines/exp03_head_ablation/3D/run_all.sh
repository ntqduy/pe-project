#!/usr/bin/env bash
# exp03_head_ablation | every 3D model of this experiment, spread over the GPU pool.
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp03_head_ablation dim:3D "$@"
