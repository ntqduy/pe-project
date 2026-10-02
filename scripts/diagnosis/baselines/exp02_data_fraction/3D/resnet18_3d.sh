#!/usr/bin/env bash
# exp02_data_fraction | 3D | ResNet-18 3D (MedicalNet): MLP head at 25/50/75/100% of the training patients (FRACTIONS=25 for one)
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp02_data_fraction resnet18_3d "$@"
