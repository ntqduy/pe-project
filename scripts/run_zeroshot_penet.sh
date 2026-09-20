#!/usr/bin/env bash
# Zero-shot PENet: run the released CTPA PE classifier over a built cohort, unchanged.
#
#   CHECK_SLICE_ORDER=1 bash scripts/run_zeroshot_penet.sh   # do this FIRST, see below
#   bash scripts/run_zeroshot_penet.sh                       # full test split
#   MAX_CASES=50 bash scripts/run_zeroshot_penet.sh          # quick look
#   PROFILE=smoke_30 bash scripts/run_zeroshot_penet.sh
#   RESTRICT_TO=<dataset>/clinical/spesi_evaluable.csv bash scripts/run_zeroshot_penet.sh
#
# Nothing is trained. The weights are loaded and applied, so this is the external
# reference an in-house imaging arm has to beat. Contract and the two repository
# gotchas it depends on: third_party/versions.yaml (components.penet).
#
# It reads the raw release NIfTI, not volumes/*.npy: PENet needs 208x208 slices at their
# original resolution and the shared cache is 1.5 mm isotropic fitted to 128^3.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-full_inspect}"
SLICE_ORDER="${SLICE_ORDER:-superior_to_inferior}"
AGGREGATE="${AGGREGATE:-max}"
PYTHON="${PYTHON:-python3}"
CONFIG="$PROJECT_ROOT/configs/runs/01_foundation/penet_zero_shot.yaml"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/third_party/weights/penet_best.pth.tar}"

if [[ -z "${PE_CLOUD_ROOT:-}" ]]; then
  printf 'error: PE_CLOUD_ROOT is not set; see scripts/README.md for the roots to export\n' >&2
  exit 2
fi
RAW_ROOT="${PE_RAW_INSPECT_ROOT:-$PE_CLOUD_ROOT/data/Stanford_INSPECT_dataset}"
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
                   --aggregate "$AGGREGATE")
[[ -n "${RESTRICT_TO:-}" ]] && common+=(--restrict-to "$RESTRICT_TO")
[[ -n "${GPUS:-}" ]] && common+=(--gpus "$GPUS")

# A 3-D convolution is sensitive to slice order inside a window, and the PENet repository
# does not record which direction its pre-sorted volumes used. Score a small sample both
# ways once: the wrong direction lands near chance. Do this before the full run.
if [[ -n "${CHECK_SLICE_ORDER:-}" ]]; then
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
[[ -n "${OVERWRITE:-}" ]] && cmd+=(--overwrite)

printf '==> zero-shot PENet  [profile=%s slices=%s agg=%s]\n' "$PROFILE" "$SLICE_ORDER" "$AGGREGATE"
printf '    weights  %s\n' "$CHECKPOINT"
printf '    raw      %s/CT/full/CTPA\n' "$RAW_ROOT"
exec "${cmd[@]}"
