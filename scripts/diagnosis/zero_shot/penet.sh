#!/usr/bin/env bash
# Zero-shot PENet: run the released CTPA PE classifier over a built cohort, unchanged.
#
#   CHECK_SLICE_ORDER=1 bash scripts/diagnosis/zero_shot/penet.sh   # do this FIRST, see below
#   bash scripts/diagnosis/zero_shot/penet.sh                       # full test split
#   MAX_CASES=50 bash scripts/diagnosis/zero_shot/penet.sh          # quick look
#   PROFILE=smoke_30 bash scripts/diagnosis/zero_shot/penet.sh
#   RESTRICT_TO=<derived>/cache/<profile>/clinical/spesi_evaluable.csv bash scripts/diagnosis/zero_shot/penet.sh
#
# Nothing is trained. The weights are loaded and applied, so this is the external
# reference an in-house imaging arm has to beat. Contract and the two repository
# gotchas it depends on: third_party/versions.yaml (components.penet).
#
# It reads the raw release NIfTI, not volumes/*.npy: PENet needs 208x208 slices at their
# original resolution and the shared cache is 1.5 mm isotropic fitted to 128^3.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../tool/use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
# shellcheck source=../../tool/_flags.sh
source "$PROJECT_ROOT/scripts/tool/_flags.sh"
PROFILE="${PROFILE:-full_inspect}"
SLICE_ORDER="${SLICE_ORDER:-superior_to_inferior}"
AGGREGATE="${AGGREGATE:-max}"
GPUS="${GPUS-0}"  # GPUS= (empty) means CPU; only an unset GPUS defaults to GPU 0
PYTHON="${PYTHON:-python3}"
CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/zero_shot/penet.yaml"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/third_party/weights/penet_best.pth.tar}"

RAW_ROOT="$PE_RAW_INSPECT_ROOT"
[[ -d "$RAW_ROOT/CT/full/CTPA" ]] || {
  printf 'error: raw CTPA series not found at %s/CT/full/CTPA\n' "$RAW_ROOT" >&2
  printf '       PENet reads the raw NIfTI, not the derived cache\n' >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || {
  printf 'error: PENet checkpoint not found: %s\n' "$CHECKPOINT" >&2; exit 2; }
# The transform depends on cv2.INTER_AREA; approximating that resize would move the score,
# so a missing opencv is a hard stop rather than a silent fallback.
"$PYTHON" -c 'import cv2' 2>/dev/null || {
  printf 'error: opencv is required (cv2.INTER_AREA slice resize)\n' >&2
  printf '       pip install opencv-python-headless\n' >&2; exit 2; }

declare -a common=(--config "$CONFIG" --set "data.profile=$PROFILE" --checkpoint "$CHECKPOINT"
                   --aggregate "$AGGREGATE" --gpus "$GPUS")
[[ -n "${RESTRICT_TO:-}" ]] && common+=(--restrict-to "$RESTRICT_TO")

# A 3-D convolution is sensitive to slice order inside a window, and the PENet repository
# does not record which direction its pre-sorted volumes used. Score a small sample both
# ways once: the wrong direction lands near chance. Do this before the full run.
if is_true CHECK_SLICE_ORDER; then
  printf '==> slice-order check  [profile=%s cases=%s]\n' "$PROFILE" "${MAX_CASES:-50}"
  for order in superior_to_inferior inferior_to_superior; do
    printf -- '--- %s\n' "$order"
    "$PYTHON" "$PROJECT_ROOT/tools/tasks/zeroshot_penet.py" "${common[@]}" \
      --slice-order "$order" --max-cases "${MAX_CASES:-50}" \
      --set "experiment.id=DX_zeroshot_penet_check_${order}" --overwrite || exit 1
  done
  printf '\nKeep the order with the higher AUROC; near 0.5 means that direction is wrong.\n'
  printf 'Then re-run without CHECK_SLICE_ORDER, passing SLICE_ORDER=<winner>.\n'
  exit 0
fi

declare -a cmd=("$PYTHON" "$PROJECT_ROOT/tools/tasks/zeroshot_penet.py" "${common[@]}"
                --slice-order "$SLICE_ORDER")
if [[ -n "${MAX_CASES:-}" ]]; then cmd+=(--max-cases "$MAX_CASES"); else cmd+=(--allow-full); fi
if is_true OVERWRITE; then cmd+=(--overwrite); fi

printf '==> zero-shot PENet  [profile=%s slices=%s agg=%s gpus=%s]\n' "$PROFILE" "$SLICE_ORDER" "$AGGREGATE" "${GPUS:-cpu}"
printf '    weights  %s\n' "$CHECKPOINT"
printf '    raw      %s/CT/full/CTPA\n' "$RAW_ROOT"
exec "${cmd[@]}"
