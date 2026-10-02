#!/usr/bin/env bash
# Zero-shot RADAR: run the released DAMO abdominal-CT generalist over a built cohort, unchanged.
#
#   bash scripts/diagnosis/zero_shot/radar.sh                       # full validation + test splits
#   MAX_CASES=20 PROFILE=smoke_30 bash scripts/diagnosis/zero_shot/radar.sh   # quick look
#   RESUME=1 bash scripts/diagnosis/zero_shot/radar.sh              # continue an interrupted run
#   RESTRICT_TO=<derived>/cache/<profile>/clinical/spesi_evaluable.csv bash scripts/diagnosis/zero_shot/radar.sh
#
# Nothing is trained. RADAR ships no PE finding, so the run scores RADAR's own pulmonary-
# artery organ token against the text pair in configs/runs/01_foundation/zero_shot/radar.yaml
# (radar.prompts): an out-of-scope probe, not a PE model. Contract and caveats:
# third_party/versions.yaml (components.radar).
#
# Two Python environments: PYTHON (the project env) selects studies and evaluates;
# RADAR_PYTHON (an env built from third_party/repos/damo-radar/requirements.txt, which pins
# transformers==4.25) runs the model. It reads the raw release NIfTI, not volumes/*.npy.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
# shellcheck source=../../tool/_flags.sh
source "$PROJECT_ROOT/scripts/tool/_flags.sh"
PROFILE="${PROFILE:-full_inspect}"
GPUS="${GPUS-0}"  # GPUS= (empty) means CPU; only an unset GPUS defaults to GPU 0
PYTHON="${PYTHON:-python3}"
CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/zero_shot/radar.yaml"
REPO="$PROJECT_ROOT/third_party/repos/damo-radar"
CHECKPOINT="${CHECKPOINT:-$REPO/ckpt/checkpoint_radar_pretrain.pth}"
if [[ -z "${RADAR_PYTHON:-}" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
  RADAR_PYTHON="$CONDA_BASE/envs/radar/bin/python"
fi

RAW_ROOT="$PE_RAW_INSPECT_ROOT"
[[ -d "$RAW_ROOT/CT/full/CTPA" ]] || {
  printf 'error: raw CTPA series not found at %s/CT/full/CTPA\n' "$RAW_ROOT" >&2
  printf '       RADAR reads the raw NIfTI, not the derived cache\n' >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || {
  printf 'error: RADAR checkpoint not found: %s\n' "$CHECKPOINT" >&2
  printf '       cd %s/download_scripts && python download_checkpoints.py\n' "$REPO" >&2; exit 2; }
[[ -x "$RADAR_PYTHON" ]] || {
  printf 'error: RADAR env python not found: %s\n' "$RADAR_PYTHON" >&2
  printf '       conda create -n radar python=3.10 && conda activate radar && pip install -r %s/requirements.txt\n' "$REPO" >&2
  printf '       or point RADAR_PYTHON at an existing env\n' >&2; exit 2; }
"$RADAR_PYTHON" -c 'import monai, nibabel, transformers, matplotlib' 2>/dev/null || {
  printf 'error: %s lacks the RADAR requirements (monai, nibabel, transformers==4.25, matplotlib)\n' "$RADAR_PYTHON" >&2
  exit 2; }
# PYTHON must be the project env, not the RADAR env: an active `conda activate radar` makes
# python3 resolve to RADAR's Python 3.10, which has numpy/yaml but cannot run project code.
( cd "$PROJECT_ROOT" && "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11); import source.metrics.result_table, tools._common' ) 2>/dev/null || {
  printf 'error: %s (%s) is not the project env (needs Python >= 3.11 and the project packages)\n' \
    "$PYTHON" "$("$PYTHON" -c 'import sys; print(sys.executable)' 2>/dev/null || echo '?')" >&2
  printf '       run conda activate pe, or set PYTHON=<pe env>/bin/python\n' >&2; exit 2; }

declare -a cmd=("$PYTHON" "$PROJECT_ROOT/tools/tasks/zeroshot_radar.py" --config "$CONFIG"
                --set "data.profile=$PROFILE" --checkpoint "$CHECKPOINT" --radar-python "$RADAR_PYTHON"
                --gpus "$GPUS")
[[ -n "${RESTRICT_TO:-}" ]] && cmd+=(--restrict-to "$RESTRICT_TO")
if [[ -n "${MAX_CASES:-}" ]]; then cmd+=(--max-cases "$MAX_CASES"); else cmd+=(--allow-full); fi
if is_true OVERWRITE; then cmd+=(--overwrite); fi
if is_true RESUME; then cmd+=(--resume); fi

printf '==> zero-shot RADAR  [profile=%s gpus=%s]\n' "$PROFILE" "$GPUS"
printf '    weights  %s\n' "$CHECKPOINT"
printf '    radar    %s\n' "$RADAR_PYTHON"
printf '    raw      %s/CT/full/CTPA\n' "$RAW_ROOT"
exec "${cmd[@]}"
