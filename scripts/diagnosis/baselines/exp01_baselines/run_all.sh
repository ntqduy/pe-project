#!/usr/bin/env bash
# exp01_baselines | all 20 baselines (2D / 2.5D / 3D), MLP head, 100% train
# Every case of the experiment (grid: experiment.yaml), spread over the GPU pool, then the
# summary tables / plots in <outputs>/diagnosis/BASE/<profile>/diagnosis/exp01_baselines/.
# Re-running resumes: finished cases are skipped.
#   GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp01_baselines/run_all.sh
#   GPUS=0,1 JOBS_PER_GPU=2 FOLDS="0 1 2 3 4" SEEDS="42 43" bash scripts/diagnosis/baselines/exp01_baselines/run_all.sh
#   DRY_LIST=1 bash scripts/diagnosis/baselines/exp01_baselines/run_all.sh                # print the grid only
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../tool/run_baseline_grid.sh" exp01_baselines all "$@"
