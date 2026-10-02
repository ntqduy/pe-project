#!/usr/bin/env bash
# exp02_data_fraction | 3D | CT-FM 3D frozen (cached features) + head: MLP head at 25/50/75/100% of the training patients (FRACTIONS=25 for one)
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp02_data_fraction ctfm_frozen_3d "$@"
