#!/usr/bin/env bash
# exp02_data_fraction | 100% of the training patients | all 8 models, MLP head
# 100% is the full official train split: the same runs as exp01_baselines (MLP head), so a
# finished exp01 run is skipped here, not trained again.
#   bash scripts/diagnosis/baselines/exp02_data_fraction/frac100/run_all.sh --gpus 0,1 --seeds "0 1 2"
#   bash scripts/diagnosis/baselines/exp02_data_fraction/frac100/run_all.sh --dry-run        # list cases + commands, run nothing
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp02_data_fraction all --fractions 100 "$@"
