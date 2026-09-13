#!/usr/bin/env bash
# Ablation 1 — architecture ablation, six variants run in sequence.
#
#   bash scripts/run_ablation_arch.sh
#   GPUS=0 DATASET=full_inspect bash scripts/run_ablation_arch.sh
#   ACTION=preflight bash scripts/run_ablation_arch.sh          # check all six, train none
#   VARIANTS="global_only full_moe" bash scripts/run_ablation_arch.sh
#   CV=1 bash scripts/run_ablation_arch.sh                      # K-fold instead of hold-out
#
# Environment (all optional):
#   VARIANTS   subset to run   default: all six, in proposal order
#   DATASET    smoke_30 | test_500_sample | full_inspect        default: test_500_sample
#   GPUS       physical GPU ids
#   ACTION     show | plan | preflight | dry | run              default: run
#   SET        extra config overrides
#   CV=1       run each variant through scripts/run_cv.sh instead of a single hold-out
#   FOLDS      folds when CV=1                                  default: config value
#   NO_EVAL=1  skip the evaluate step (metrics + CI will be missing)
#   KEEP_GOING=1  continue after a failing variant
#
# Every variant retrains from C0 and changes only task.regions and the fusion component, so
# a difference between rows is a difference in architecture and nothing else.
# Output: <output_root>/ablation/architecture/AB_arch_<variant>/
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/_lib.sh"

CONFIG_DIR="configs/runs/05_anatomy_analysis/architecture"
DEFAULT_VARIANTS="global_only global_heart global_pa global_lung full_moe full_no_router"
read -r -a variants <<< "${VARIANTS:-$DEFAULT_VARIANTS}"

failed=()
for variant in "${variants[@]}"; do
  config="$PROJECT_ROOT/$CONFIG_DIR/$variant.yaml"
  if [[ ! -f "$config" ]]; then
    printf 'error: unknown architecture variant %q (no %s)\n' "$variant" "$CONFIG_DIR/$variant.yaml" >&2
    exit 2
  fi
  printf '\n=== architecture ablation: %s ===\n' "$variant"

  if [[ -n "${CV:-}" ]]; then
    if FOLDS="${FOLDS:-}" GPUS="${GPUS:-}" SET="${SET:-}" \
       bash "$PROJECT_ROOT/scripts/run_cv.sh" "$CONFIG_DIR/$variant.yaml"; then
      continue
    fi
    failed+=("$variant")
    [[ -n "${KEEP_GOING:-}" ]] || exit 1
    continue
  fi

  status=0
  pe_run "ablation.arch.$variant" || status=$?
  if (( status != 0 )); then
    failed+=("$variant")
    [[ -n "${KEEP_GOING:-}" ]] || exit "$status"
    continue
  fi

  if [[ -z "${NO_EVAL:-}" && "${ACTION:-run}" == "run" ]]; then
    declare -a evaluate=(--config "$CONFIG_DIR/$variant.yaml" --allow-full)
    evaluate+=(--set "data.profile=${DATASET:-test_500_sample}")
    if [[ -n "${SET:-}" ]]; then
      # shellcheck disable=SC2206 - word splitting is the documented interface
      extra=(${SET})
      for override in "${extra[@]}"; do evaluate+=(--set "$override"); done
    fi
    "${PYTHON:-python}" "$PROJECT_ROOT/tools/tasks/evaluate.py" "${evaluate[@]}" || {
      failed+=("$variant:evaluate")
      [[ -n "${KEEP_GOING:-}" ]] || exit 1
    }
  fi
done

if (( ${#failed[@]} )); then
  printf '\narchitecture ablation finished with failures: %s\n' "${failed[*]}" >&2
  exit 1
fi
printf '\narchitecture ablation complete: %s\n' "${variants[*]}"
