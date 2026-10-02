#!/usr/bin/env bash
# exp02_data_fraction | 8 3D models x 25/50/75/100% training patients, MLP head
# Every case of the experiment (grid: experiment.yaml), spread over the GPU pool, then the
# summary tables / plots in <outputs>/diagnosis/BASE/<profile>/diagnosis/exp02_data_fraction/.
# Re-running resumes: finished cases are skipped.
#   GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh
#   GPUS=0,1 JOBS_PER_GPU=2 FOLDS="0 1 2 3 4" SEEDS="42 43" bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh
#   DRY_LIST=1 bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh                # print the grid only
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../tool/run_baseline_grid.sh" exp02_data_fraction all "$@"
