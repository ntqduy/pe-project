#!/usr/bin/env bash
# exp03_head_ablation | 3D | CT-FM 3D + LoRA: MLP vs KAN head (HEADS=kan for one)
# Settings are environment variables, documented in scripts/tool/run_baseline_grid.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../../tool/run_baseline_grid.sh" exp03_head_ablation ctfm_lora_3d "$@"
