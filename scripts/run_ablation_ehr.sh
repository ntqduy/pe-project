#!/usr/bin/env bash
# EHR variable ablation — four variants run in sequence.
#
#   bash scripts/run_ablation_ehr.sh
#   ACTION=preflight bash scripts/run_ablation_ehr.sh
#   VARIANTS="all_with_missingness all_no_missingness" bash scripts/run_ablation_ehr.sh
#   EHR_COLUMNS="[age,sex,hr,sbp]" bash scripts/run_ablation_ehr.sh   # điền danh sách thật
#   CV=1 bash scripts/run_ablation_ehr.sh
#
# Environment (all optional):
#   VARIANTS      subset to run   default: all four
#   EHR_COLUMNS   value for data.ehr_columns on the `all_*` variants
#   EHR_COLUMNS_COMMON  value for data.ehr_columns on the `common_*` variants
#   DATASET GPUS ACTION SET CV FOLDS NO_EVAL KEEP_GOING   same as run_ablation_arch.sh
#
# Hai trục đang đổi: tập biến (all / common) × missingness indicator (có / không).
# `data.ehr_columns` và `data.ehr_columns_common` còn rỗng trong repo, nên phải truyền qua
# EHR_COLUMNS / EHR_COLUMNS_COMMON hoặc điền thẳng vào components/tasks.yaml trước khi chạy.
# Output: <output_root>/ablation/ehr/AB_ehr_<variant>/
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/_lib.sh"

CONFIG_DIR="configs/runs/04_prognosis/ehr_ablation"
DEFAULT_VARIANTS="all_with_missingness all_no_missingness common_with_missingness common_no_missingness"
read -r -a variants <<< "${VARIANTS:-$DEFAULT_VARIANTS}"

failed=()
for variant in "${variants[@]}"; do
  config="$PROJECT_ROOT/$CONFIG_DIR/$variant.yaml"
  if [[ ! -f "$config" ]]; then
    printf 'error: unknown EHR variant %q (no %s)\n' "$variant" "$CONFIG_DIR/$variant.yaml" >&2
    exit 2
  fi

  columns=""
  case "$variant" in
    common_*) columns="${EHR_COLUMNS_COMMON:-}" ;;
    *)        columns="${EHR_COLUMNS:-}" ;;
  esac
  variant_set="${SET:-}"
  [[ -n "$columns" ]] && variant_set="${variant_set:+$variant_set }data.ehr_columns=$columns"

  printf '\n=== EHR ablation: %s ===\n' "$variant"

  if [[ -n "${CV:-}" ]]; then
    if FOLDS="${FOLDS:-}" GPUS="${GPUS:-}" SET="$variant_set" \
       bash "$PROJECT_ROOT/scripts/run_cv.sh" "$CONFIG_DIR/$variant.yaml"; then
      continue
    fi
    failed+=("$variant"); [[ -n "${KEEP_GOING:-}" ]] || exit 1; continue
  fi

  status=0
  SET="$variant_set" pe_run "ablation.ehr.$variant" || status=$?
  if (( status != 0 )); then
    failed+=("$variant"); [[ -n "${KEEP_GOING:-}" ]] || exit "$status"; continue
  fi

  if [[ -z "${NO_EVAL:-}" && "${ACTION:-run}" == "run" ]]; then
    declare -a evaluate=(--config "$CONFIG_DIR/$variant.yaml" --allow-full)
    evaluate+=(--set "data.profile=${DATASET:-test_500_sample}")
    if [[ -n "$variant_set" ]]; then
      # shellcheck disable=SC2206 - word splitting is the documented interface
      extra=($variant_set)
      for override in "${extra[@]}"; do evaluate+=(--set "$override"); done
    fi
    "${PYTHON:-python}" "$PROJECT_ROOT/tools/tasks/evaluate.py" "${evaluate[@]}" || {
      failed+=("$variant:evaluate"); [[ -n "${KEEP_GOING:-}" ]] || exit 1
    }
  fi
done

if (( ${#failed[@]} )); then
  printf '\nEHR ablation finished with failures: %s\n' "${failed[*]}" >&2
  exit 1
fi
printf '\nEHR ablation complete: %s\n' "${variants[*]}"
