#!/usr/bin/env bash
# exp04_slice_ablation | 2.5D | ResNet-18 2.5D slice-MIL: every variant (VARIANTS / --variants for a subset)
# Settings: scripts/tool/run_baseline_grid.sh (flags or environment variables).
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp04_slice_ablation resnet18_25d "$@"
