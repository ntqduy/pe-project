#!/usr/bin/env bash
# TotalSegmentator pseudo-anatomy masks, with LungMask contributing an independent lung Dice cross-check.
#
# Thin wrapper: resolves data.segmentation through run.py. No scientific logic lives here.
#
#   MAX_CASES=10 GPUS=0 bash scripts/1_segmentation/totalsegmentator.sh   # smoke, one GPU
#   ALLOW_ALL=1 GPUS=0,1 bash scripts/1_segmentation/totalsegmentator.sh  # study sharding
#   ALLOW_ALL=1 SET="segmentation.workers=8" bash scripts/1_segmentation/totalsegmentator.sh
#
# Per-study parallelism and previews are configured by segmentation.workers / preview_*
# in configs/runs/00_data/segmentation.yaml. Terminal output is captured in logs/run.log.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run --scoped data.segmentation "$@"
