#!/usr/bin/env bash
# exp02_data_fraction | 75% of the training patients | all 8 models, MLP head
#   bash scripts/diagnosis/baselines/exp02_data_fraction/frac075/run_all.sh --gpus 0,1 --seeds "0 1 2"
#   bash scripts/diagnosis/baselines/exp02_data_fraction/frac075/run_all.sh --dry-run        # list cases + commands, run nothing
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp02_data_fraction all --fractions 75 "$@"
