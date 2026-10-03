#!/usr/bin/env bash
# exp02_data_fraction | 100% of the training patients | Swin 3D (Swin-UNETR self-supervised CT) | MLP head
# 100% is the full official train split: the same runs as exp01_baselines (MLP head), so a
# finished exp01 run is skipped here, not trained again.
# Subsets: by patient, stratified by label, nested 25 c 50 c 75 c 100, drawn per seed; validation
# and test unchanged. Flags (--gpus, --seeds, --task/--label, ...): scripts/tool/run_baseline_grid.sh.
#   bash scripts/diagnosis/baselines/exp02_data_fraction/frac100/swin_3d.sh --gpus 0 --seeds "0 1 2"
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp02_data_fraction swin_3d --fractions 100 "$@"
