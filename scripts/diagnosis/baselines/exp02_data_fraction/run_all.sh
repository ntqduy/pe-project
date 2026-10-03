#!/usr/bin/env bash
# exp02_data_fraction | every fraction (25 / 50 / 75 / 100%) x 8 3D models, MLP head
# One fraction only: frac025/ frac050/ frac075/ frac100/ (each has run_all.sh + one script per model).
# The 100% runs are the exp01 runs (shared). Summary: <outputs>/<family>/BASE/<profile>/<task>/
# exp02_data_fraction/ (auroc_vs_fraction.png); subsets: <task>/splits/data_fraction/seed_<s>/.
#   bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh --gpus 0,1 --seeds "0 1 2"
#   bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh --task prognosis --label 1_month_mortality
#   bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh --dry-run
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../tool/run_baseline_grid.sh" exp02_data_fraction all "$@"
