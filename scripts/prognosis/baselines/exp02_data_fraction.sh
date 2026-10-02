#!/usr/bin/env bash
# Prognosis | exp02_data_fraction: the same grid as scripts/diagnosis/baselines/exp02_data_fraction, image-only prognosis.
# One outcome per call (required): --label 1_month_mortality | 6_month_mortality | 12_month_mortality |
#   1_month_readmission | 6_month_readmission | 12_month_readmission | 12_month_PH ; --cohort all (default) | pe
#   bash scripts/prognosis/baselines/exp02_data_fraction.sh --label 12_month_PH --gpus 0 --seeds "0 1 2"
#   for L in 1_month_mortality 12_month_PH; do bash scripts/prognosis/baselines/exp02_data_fraction.sh --label "$L"; done
# Outputs: <outputs>/prognosis/BASE/<profile>/prognosis_<cohort>_<label>/{runs,exp02_data_fraction}/
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../diagnosis/baselines/exp02_data_fraction/run_all.sh" --task prognosis "$@"
