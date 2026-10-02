#!/usr/bin/env bash
# exp03_head_ablation | 6 3D models x {MLP, KAN} head, 100% train
# Every case of the experiment (grid: experiment.yaml), spread over the GPU pool, then the
# summary tables / plots in <outputs>/<family>/BASE/<profile>/<task>/exp03_head_ablation/.
# Re-running resumes: finished cases are skipped. Flags / variables: scripts/tool/run_baseline_grid.sh.
#   bash scripts/diagnosis/baselines/exp03_head_ablation/run_all.sh --gpus 0 --seeds "0 1 2"
#   bash scripts/diagnosis/baselines/exp03_head_ablation/run_all.sh --task prognosis --label 1_month_mortality --gpus 0,1 --runs-per-gpu 2
#   bash scripts/diagnosis/baselines/exp03_head_ablation/run_all.sh --dry-run          # list cases + commands, run nothing
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../tool/run_baseline_grid.sh" exp03_head_ablation all "$@"
