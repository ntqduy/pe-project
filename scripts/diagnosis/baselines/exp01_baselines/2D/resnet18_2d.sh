#!/usr/bin/env bash
# exp01_baselines | 2D | ResNet-18 2D middle slice (ImageNet): MLP head, 100% train
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp01_baselines resnet18_2d "$@"
