#!/usr/bin/env bash
# Prognosis organ-only sufficiency — the three arms the professor's email lists under
# "Summary of prognosis experiments" (strict-heart, artery-only, lung-only) plus the
# volume-matched random control they must be read against.
#
#   bash scripts/run_prognosis_organ.sh
#   GPUS=0 DATASET=full_inspect bash scripts/run_prognosis_organ.sh
#   ACTION=preflight bash scripts/run_prognosis_organ.sh        # check all four, train none
#   ARMS="heart random" bash scripts/run_prognosis_organ.sh
#   CV=1 bash scripts/run_prognosis_organ.sh                    # K-fold instead of hold-out
#
# Environment (all optional):
#   ARMS       subset to run   default: heart pa lung random
#   DATASET    smoke_30 | test_500_sample | full_inspect        default: test_500_sample
#   GPUS       physical GPU ids
#   ACTION     show | plan | preflight | dry | run              default: run
#   SET        extra config overrides
#   CV=1       run each arm through scripts/run_cv.sh instead of a single hold-out
#   FOLDS      folds when CV=1                                  default: config value
#   NO_EVAL=1  skip the evaluate step (metrics + CI will be missing)
#   KEEP_GOING=1  continue after a failing arm
#
# Every arm is prog.image with the volume outside one ROI erased: same backbone, same C0
# initialization, same LoRA contract, same head, same cohort. The only thing that changes
# between rows is which voxels the model is allowed to see, so a gap between two rows is a
# gap in anatomy and nothing else.
#
# Read the table as: prog.image (full CT) is the ceiling, prog.random_student is the floor.
# An organ row means something only in the band between them.
#
# Output: <output_root>/roi_students/prognosis/RS_prog_<arm>_only_gt/
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/_lib.sh"

CONFIG_DIR="configs/runs/04_prognosis/students"
DEFAULT_ARMS="heart pa lung random"
read -r -a arms <<< "${ARMS:-$DEFAULT_ARMS}"

failed=()
for arm in "${arms[@]}"; do
  config="$PROJECT_ROOT/$CONFIG_DIR/$arm.yaml"
  if [[ ! -f "$config" ]]; then
    printf 'error: unknown prognosis organ arm %q (no %s)\n' "$arm" "$CONFIG_DIR/$arm.yaml" >&2
    exit 2
  fi
  printf '\n=== prognosis organ-only: %s ===\n' "$arm"

  if [[ -n "${CV:-}" ]]; then
    if FOLDS="${FOLDS:-}" GPUS="${GPUS:-}" SET="${SET:-}" \
       bash "$PROJECT_ROOT/scripts/run_cv.sh" "$CONFIG_DIR/$arm.yaml"; then
      continue
    fi
    failed+=("$arm")
    [[ -n "${KEEP_GOING:-}" ]] || exit 1
    continue
  fi

  status=0
  pe_run "prog.${arm}_student" || status=$?
  if (( status != 0 )); then
    failed+=("$arm")
    [[ -n "${KEEP_GOING:-}" ]] || exit "$status"
    continue
  fi

  if [[ -z "${NO_EVAL:-}" && "${ACTION:-run}" == "run" ]]; then
    declare -a evaluate=(--config "$CONFIG_DIR/$arm.yaml" --allow-full)
    evaluate+=(--set "data.profile=${DATASET:-test_500_sample}")
    if [[ -n "${SET:-}" ]]; then
      # shellcheck disable=SC2206 - word splitting is the documented interface
      extra=(${SET})
      for override in "${extra[@]}"; do evaluate+=(--set "$override"); done
    fi
    "${PYTHON:-python}" "$PROJECT_ROOT/tools/tasks/evaluate.py" "${evaluate[@]}" || {
      failed+=("$arm:evaluate")
      [[ -n "${KEEP_GOING:-}" ]] || exit 1
    }
  fi
done

if (( ${#failed[@]} )); then
  printf '\nprognosis organ-only finished with failures: %s\n' "${failed[*]}" >&2
  exit 1
fi
printf '\nprognosis organ-only complete: %s\n' "${arms[*]}"
printf 'Reference row: bash scripts/5_prognosis/image_only.sh (full-CT ceiling)\n'
