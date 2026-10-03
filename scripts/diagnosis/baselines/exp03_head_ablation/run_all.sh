#!/usr/bin/env bash
# exp03_head_ablation | KAN head for all 6 3D models, 100% train
# The MLP arm is the exp01_baselines run of each model (shared): run exp01 first. The summary
# (<outputs>/<family>/BASE/<profile>/<task>/exp03_head_ablation/, mlp_vs_kan.png) pairs both heads.
#   bash scripts/diagnosis/baselines/exp03_head_ablation/run_all.sh --gpus 0 --seeds "0 1 2"
#   bash scripts/diagnosis/baselines/exp03_head_ablation/run_all.sh --task prognosis --label 12_month_PH
#   bash scripts/diagnosis/baselines/exp03_head_ablation/run_all.sh --dry-run
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../tool/run_baseline_grid.sh" exp03_head_ablation all --heads kan "$@"
