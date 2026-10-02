#!/usr/bin/env bash
# Rebuild the summary tables / plots of one or all baseline experiments from finished runs.
#   bash scripts/diagnosis/baselines/summarize.sh                  # all three experiments
#   PROFILE=smoke_30 bash scripts/diagnosis/baselines/summarize.sh exp02_data_fraction
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
cd "$PROJECT_ROOT"
EXPERIMENTS=(exp01_baselines exp02_data_fraction exp03_head_ablation)
(( $# )) && EXPERIMENTS=("$@")
for NAME in "${EXPERIMENTS[@]}"; do
  case "$NAME" in
    exp01_baselines|exp02_data_fraction|exp03_head_ablation) ;;
    *)
      printf 'usage: bash scripts/diagnosis/baselines/summarize.sh [exp01_baselines|exp02_data_fraction|exp03_head_ablation ...]\n' >&2
      printf '       settings: PROFILE, TASK, TARGET, COHORT, PYTHON (got %s)\n' "$NAME" >&2
      exit 2
      ;;
  esac
  "${PYTHON:-python3}" tools/baselines/summarize.py --exp "$NAME" --profile "${PROFILE:-full_inspect}" \
    --task "${TASK:-diagnosis}" --target "${TARGET:-1_month_mortality}" --cohort "${COHORT:-all}"
done
