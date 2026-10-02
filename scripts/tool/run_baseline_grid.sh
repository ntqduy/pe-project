#!/usr/bin/env bash
# Shared driver behind scripts/diagnosis/baselines/exp0*/**.sh: one experiment; one model, one
# dimension group (dim:2D | dim:2_5D | dim:3D) or all of its models.
#
#   bash scripts/tool/run_baseline_grid.sh <experiment> <model|dim:2D|dim:2_5D|dim:3D|all> [flags] [run_case.py args]
#
# Flags (each one equals the environment variable in brackets; a flag wins):
#   --task diagnosis|prognosis [TASK]   --label <outcome> [LABEL]   --cohort all|pe [COHORT]
#   --seeds "0 1 2" [SEEDS]             --gpus 0,1 [GPUS]           --runs-per-gpu N [JOBS_PER_GPU]
#   --profile <p> [PROFILE]             --variants "center mean" [VARIANTS]   --heads / --fractions
#   --dry-run [DRY_LIST=1]              print every case and its run_case.py command, start nothing
# Anything else is passed to every run_case.py (e.g. --set key=value, --overwrite).
#
# Environment variables:
#   PROFILE=full_inspect      dataset profile (smoke_30 | test_500_sample | full_inspect)
#   GPUS=0                    GPU pool; cases are spread over it (''=CPU)
#   JOBS_PER_GPU=1            cases sharing one GPU at a time (e.g. 2-4 for ctfm_frozen_3d)
#   GPUS_PER_JOB=1            >1 = one DDP case over several GPUs
#   FOLDS=official            "official" and/or fold numbers: FOLDS="0 1 2 3 4" (k-fold, K=${K:-5})
#   SEEDS="0 1 2"             training seeds (official split, repeated with seeds)
#   VARIANTS                  subset of the experiment's variants (exp04: default center mean max)
#   HEADS / FRACTIONS         override the experiment's heads / training fractions
#   ACTION=all                all | prepare | train | evaluate | preflight | dry
#   EPOCHS / EARLY_STOPPING / BATCH_SIZE / ACCUMULATION / LR   training overrides
#   EPOCH_AUC=0               skip the per-epoch AUROC pass over train/validation (default on)
#   SCRATCH=1                 ignore pretrained weights (random init)
#   OVERWRITE=1               replace finished cases instead of skipping them
#   TASK=diagnosis            or prognosis: then LABEL is required (1_month_mortality, ...,
#                             12_month_PH) and COHORT=all|pe; LABEL with diagnosis is an error
#   NUM_WORKERS=auto  PYTHON=python3
#   RUN_MANY_EXTRA="--log-dir <dir> --no-summary"   extra tools/baselines/run_many.py options
#   DRY_LIST=1                print the case grid and commands only
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

PASSTHROUGH=()
while (( $# )); do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --cohort) COHORT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --runs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --heads) HEADS="$2"; shift 2 ;;
    --fractions) FRACTIONS="$2"; shift 2 ;;
    --dry-run) DRY_LIST=1; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done
set -- ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}

if [[ -n "${TARGET:-}" ]]; then
  printf 'error: TARGET was renamed LABEL (e.g. TASK=prognosis LABEL=%s)\n' "$TARGET" >&2
  exit 2
fi
GRID=(
  --exp "$EXPERIMENT"
  --profile "${PROFILE:-full_inspect}"
  --gpus "${GPUS-0}"
  --jobs-per-gpu "${JOBS_PER_GPU:-1}"
  --gpus-per-job "${GPUS_PER_JOB:-1}"
  --folds "${K:-5}"
  --task "${TASK:-diagnosis}"
  --cohort "${COHORT:-all}"
)
[[ -n "${LABEL:-}" ]] && GRID+=(--label "$LABEL")
# shellcheck disable=SC2206
GRID+=(--folds-to-run ${FOLDS:-official} --seeds ${SEEDS:-0 1 2})
case "$MODEL" in
  all) ;;
  dim:*) GRID+=(--dims "${MODEL#dim:}") ;;
  *) GRID+=(--models "$MODEL") ;;
esac
# shellcheck disable=SC2206
[[ -n "${HEADS:-}" ]] && GRID+=(--heads ${HEADS})
# shellcheck disable=SC2206
[[ -n "${FRACTIONS:-}" ]] && GRID+=(--fractions ${FRACTIONS})
# shellcheck disable=SC2206
[[ -n "${VARIANTS:-}" ]] && GRID+=(--variants ${VARIANTS})
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
