#!/usr/bin/env bash
# exp01_baselines | every 3D model of this experiment, spread over the GPU pool.
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp01_baselines dim:3D "$@"
