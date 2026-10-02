#!/usr/bin/env bash
# exp01_baselines | 2D | ViT-S/16 2D middle slice (ImageNet-21k): MLP head, 100% train
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp01_baselines vit_2d "$@"
