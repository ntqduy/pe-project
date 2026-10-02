#!/usr/bin/env bash
# Shared driver behind scripts/diagnosis/baselines/exp0*/**.sh: one experiment; one model, one
# dimension group (dim:2D | dim:2_5D | dim:3D) or all of its models.
#
#   bash scripts/tool/run_baseline_grid.sh <experiment> <model|dim:2D|dim:2_5D|dim:3D|all> [run_case.py args]
#
# Everything is set with environment variables in front of the command:
#   PROFILE=full_inspect      dataset profile (smoke_30 | test_500_sample | full_inspect)
#   GPUS=0                    GPU pool; cases are spread over it (''=CPU)
#   JOBS_PER_GPU=1            cases sharing one GPU at a time (e.g. 2-4 for ctfm_frozen_3d)
#   GPUS_PER_JOB=1            >1 = one DDP case over several GPUs
#   FOLDS=official            "official" and/or fold numbers: FOLDS="0 1 2 3 4" (k-fold, K=${K:-5})
#   SEEDS=42                  training seeds, e.g. SEEDS="42 43 44"
#   HEADS / FRACTIONS         override the experiment's heads / training fractions
#   ACTION=all                all | prepare | train | evaluate | preflight | dry
#   EPOCHS / EARLY_STOPPING / BATCH_SIZE / ACCUMULATION / LR   training overrides
#   EPOCH_AUC=0               skip the per-epoch AUROC pass over train/validation (default on)
#   SCRATCH=1                 ignore pretrained weights (random init)
#   OVERWRITE=1               replace finished cases instead of skipping them
#   TASK=diagnosis            or prognosis (TARGET=1_month_mortality, COHORT=all|pe)
#   NUM_WORKERS=auto  PYTHON=python3
#   RUN_MANY_EXTRA="--log-dir <dir> --no-summary"   extra tools/baselines/run_many.py options
#   DRY_LIST=1                print the case grid only
# Boolean flags accept 1/0, true/false, yes/no, on/off; anything else is an error.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=use_gcs_storage.sh
source "$PROJECT_ROOT/scripts/tool/use_gcs_storage.sh"
# shellcheck source=_flags.sh
source "$PROJECT_ROOT/scripts/tool/_flags.sh"
EXPERIMENT="${1:?usage: run_baseline_grid.sh <experiment> <model|all> [run_case args]}"
MODEL="${2:?usage: run_baseline_grid.sh <experiment> <model|all> [run_case args]}"
shift 2
PYTHON="${PYTHON:-python3}"

GRID=(
  --exp "$EXPERIMENT"
  --profile "${PROFILE:-full_inspect}"
  --gpus "${GPUS-0}"
  --jobs-per-gpu "${JOBS_PER_GPU:-1}"
  --gpus-per-job "${GPUS_PER_JOB:-1}"
  --folds "${K:-5}"
  --task "${TASK:-diagnosis}"
  --target "${TARGET:-1_month_mortality}"
  --cohort "${COHORT:-all}"
)
# shellcheck disable=SC2206
GRID+=(--folds-to-run ${FOLDS:-official} --seeds ${SEEDS:-42})
case "$MODEL" in
  all) ;;
  dim:*) GRID+=(--dims "${MODEL#dim:}") ;;
  *) GRID+=(--models "$MODEL") ;;
esac
# shellcheck disable=SC2206
[[ -n "${HEADS:-}" ]] && GRID+=(--heads ${HEADS})
# shellcheck disable=SC2206
[[ -n "${FRACTIONS:-}" ]] && GRID+=(--fractions ${FRACTIONS})
if is_true DRY_LIST; then GRID+=(--dry-list); fi

CASE=(--action "${ACTION:-all}" --num-workers "${NUM_WORKERS:-auto}" --python "$PYTHON")
[[ -n "${EPOCHS:-}" ]] && CASE+=(--epochs "$EPOCHS")
[[ -n "${EARLY_STOPPING:-}" ]] && CASE+=(--patience "$EARLY_STOPPING")
[[ -n "${BATCH_SIZE:-}" ]] && CASE+=(--batch-size "$BATCH_SIZE")
[[ -n "${ACCUMULATION:-}" ]] && CASE+=(--accumulation "$ACCUMULATION")
[[ -n "${LR:-}" ]] && CASE+=(--lr "$LR")
if is_false EPOCH_AUC; then CASE+=(--no-epoch-auc); fi
if is_true SCRATCH; then CASE+=(--scratch); fi
if is_true OVERWRITE; then CASE+=(--overwrite); fi

cd "$PROJECT_ROOT"
exec "$PYTHON" tools/baselines/run_many.py ${RUN_MANY_EXTRA:-} "${GRID[@]}" -- "${CASE[@]}" "$@"
