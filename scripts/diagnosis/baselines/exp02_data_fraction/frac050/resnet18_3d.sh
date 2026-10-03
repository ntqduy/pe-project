#!/usr/bin/env bash
# exp02_data_fraction | 50% of the training patients | ResNet-18 3D (MedicalNet) | MLP head
# Subsets: by patient, stratified by label, nested 25 c 50 c 75 c 100, drawn per seed; validation
# and test unchanged. Flags (--gpus, --seeds, --task/--label, ...): scripts/tool/run_baseline_grid.sh.
#   bash scripts/diagnosis/baselines/exp02_data_fraction/frac050/resnet18_3d.sh --gpus 0 --seeds "0 1 2"
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp02_data_fraction resnet18_3d --fractions 50 "$@"
