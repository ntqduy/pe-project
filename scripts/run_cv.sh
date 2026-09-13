#!/usr/bin/env bash
# Patient-level stratified K-fold cross-validation around the unchanged task trainer.
#
#   bash scripts/run_cv.sh configs/runs/03_diagnosis/matrix/single_task.yaml
#   FOLDS=3 GPUS=0 bash scripts/run_cv.sh configs/runs/04_prognosis/modality/image_ehr_pesi.yaml
#   DRY_RUN=1 bash scripts/run_cv.sh configs/runs/03_diagnosis/baseline/global_single.yaml
#
# Environment (all optional):
#   FOLDS       override evaluation.cross_validation.folds
#   GPUS        physical GPU ids, e.g. 0 or 0,1
#   SET         extra config overrides, e.g. SET="encoder.init_source=silver"
#   DRY_RUN=1   write fold manifests and print the plan; train nothing
#   OVERWRITE=1 discard an existing per-fold run instead of resuming
#
# The config must enable CV (evaluation.cross_validation.enabled=true) or FOLDS must be set.
# Output: <output_root>/cross_validation/<family>/<experiment_id>/
#   folds/fold_<k>/manifest.csv · fold_plan.json · cv_summary.json · CV_SUMMARY.md · logs/run.log
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PE_CLOUD_ROOT:-}" && -f "$PROJECT_ROOT/scripts/use_gcs_storage.sh" ]]; then
  # shellcheck disable=SC1091 -- intentional repository-local environment helper
  source "$PROJECT_ROOT/scripts/use_gcs_storage.sh"
fi
PYTHON="${PYTHON:-python}"

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
  printf 'usage: bash scripts/run_cv.sh <config.yaml>\n' >&2
  exit 2
fi
shift || true

declare -a args=(--config "$CONFIG")
[[ -n "${FOLDS:-}" ]] && args+=(--folds "$FOLDS")
[[ -n "${GPUS:-}" ]] && args+=(--gpus "$GPUS")
[[ -n "${DRY_RUN:-}" ]] && args+=(--dry-run)
[[ -n "${OVERWRITE:-}" ]] && args+=(--overwrite)
if [[ -n "${SET:-}" ]]; then
  # shellcheck disable=SC2206 - word splitting is the documented interface
  extra=(${SET})
  for override in "${extra[@]}"; do args+=(--set "$override"); done
fi

printf '==> cross-validation  [config=%s folds=%s]\n' "$CONFIG" "${FOLDS:-config}"
exec "$PYTHON" "$PROJECT_ROOT/tools/tasks/train_cv.py" "${args[@]}" "$@"
