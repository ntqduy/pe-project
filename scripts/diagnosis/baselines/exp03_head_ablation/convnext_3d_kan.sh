#!/usr/bin/env bash
# exp03_head_ablation | ConvNeXt-T 3D (ImageNet inflated) | KAN head, 100% train
# The MLP arm of the comparison is the exp01_baselines run of the same model (shared); run
# exp01 first, then this, and summarize.sh exp03_head_ablation compares MLP vs KAN.
#   bash scripts/diagnosis/baselines/exp03_head_ablation/convnext_3d_kan.sh --gpus 0 --seeds "0 1 2"
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../tool/run_baseline_grid.sh" exp03_head_ablation convnext_3d --heads kan "$@"
