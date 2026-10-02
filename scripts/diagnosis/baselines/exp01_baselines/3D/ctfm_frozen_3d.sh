#!/usr/bin/env bash
# exp01_baselines | 3D | CT-FM 3D frozen (cached features) + head: MLP head, 100% train
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp01_baselines ctfm_frozen_3d "$@"
