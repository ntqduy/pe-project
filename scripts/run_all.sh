#!/usr/bin/env bash
# Enumerate (and optionally execute) the protocol experiment matrix.
#
# This script contains no scientific logic and no config of its own: it only iterates the
# axes and invokes the same per-experiment launchers a human would run by hand, so a
# matrix run and a single run go through identical code.
#
# It PRINTS the matrix and exits by default. The full matrix is thousands of trainings,
# so executing it is an explicit opt-in:
#
#   bash scripts/run_all.sh                       # list the plan and the run count
#   EXECUTE=1 ACTION=preflight bash scripts/run_all.sh   # check every run without training
#   EXECUTE=1 bash scripts/run_all.sh             # actually train the matrix
#
# Axes (space-separated; each defaults to the full protocol set):
#   STAGES          rspect diagnosis prognosis silver_encoder
#   WEIGHT_SOURCES  pretrained dapt alignment rspect_multitask rspect_single silver_encoder
#   COHORTS         all_comers pe_positive_only (legacy names remain accepted)
#   EHR_PROFILES    EHR_0_h EHR_24_h
#   TASKS           the seven prognosis outcomes
#   STRATEGIES      the thirteen prognosis strategies
#   DX_MODES        multi_task single_task
#   DX_LABELS       pe_positive pe_acute pe_subsegmental
#   DX_STRATEGIES   global probe
#
# Everything else (DATASET, ACTION, GPUS, SMOKE, SET, RESUME, OVERWRITE) is passed
# through to the launchers untouched.
#
#   KEEP_GOING=1    continue after a failing run instead of stopping at the first one
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGES="${STAGES:-rspect diagnosis prognosis}"
WEIGHT_SOURCES="${WEIGHT_SOURCES:-pretrained dapt alignment rspect_multitask rspect_single silver_encoder}"
COHORTS="${COHORTS:-all_comers pe_positive_only}"
EHR_PROFILES="${EHR_PROFILES:-EHR_0_h EHR_24_h}"
TASKS="${TASKS:-1_month_mortality 6_month_mortality 12_month_mortality 1_month_readmission 6_month_readmission 12_month_readmission 12_month_PH}"
STRATEGIES="${STRATEGIES:-image_only clinical_only pesi_only clinical_pesi image_clinical image_pesi image_clinical_pesi anatomy_concat anatomy_late_logit anatomy_soft_moe global_late_logit global_soft_moe probe}"
DX_MODES="${DX_MODES:-multi_task single_task}"
DX_LABELS="${DX_LABELS:-pe_positive pe_acute pe_subsegmental}"
DX_STRATEGIES="${DX_STRATEGIES:-global probe}"
SILVER_LABELS="${SILVER_LABELS:-medgemma rule_falcon rule_falcon_medgemma}"

# RSPECT is an upstream stage: it may only start from a stage that precedes it.
RSPECT_WEIGHT_SOURCES="${RSPECT_WEIGHT_SOURCES:-pretrained dapt alignment}"

# Strategies that read no image. Their model is identical for every WEIGHT_SOURCE, so the
# matrix emits them once instead of training the same tabular model six times over.
TABULAR_ONLY=" clinical_only pesi_only clinical_pesi "
TABULAR_WEIGHT_SOURCE="${TABULAR_WEIGHT_SOURCE:-alignment}"
# ... but never silently drop them: if the chosen tag is not among the selected weight
# sources, attribute them to the first source that is.
if [[ " $WEIGHT_SOURCES " != *" $TABULAR_WEIGHT_SOURCE "* ]]; then
  TABULAR_WEIGHT_SOURCE="${WEIGHT_SOURCES%% *}"
fi

planned=0
failed=0
declare -a FAILED_RUNS=()

emit() {
  # emit <script> [VAR=VALUE ...]
  local script="$1"; shift
  local -a env_pairs=("$@")
  planned=$((planned + 1))
  printf '%s' "[$planned]"
  local pair
  for pair in "${env_pairs[@]}"; do printf ' %s' "$pair"; done
  printf ' bash %s\n' "${script#"$ROOT/"}"
  if [[ -n "${EXECUTE:-}" ]]; then
    if env "${env_pairs[@]}" bash "$script"; then
      return 0
    fi
    failed=$((failed + 1))
    FAILED_RUNS+=("${env_pairs[*]} bash ${script#"$ROOT/"}")
    if [[ -z "${KEEP_GOING:-}" ]]; then
      printf 'error: run failed; stopping. Set KEEP_GOING=1 to continue past failures.\n' >&2
      exit 1
    fi
  fi
}

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if has_stage rspect; then
  for weight in $RSPECT_WEIGHT_SOURCES; do
    emit "$ROOT/3_shared_encoder/3_rspect/multi-task/train.sh" "WEIGHT_SOURCE=$weight"
    emit "$ROOT/3_shared_encoder/3_rspect/single-task/pe_positive.sh" "WEIGHT_SOURCE=$weight"
  done
fi

if has_stage silver_encoder; then
  for weight in $WEIGHT_SOURCES; do
    [[ "$weight" == "silver_encoder" ]] && continue
    for label in $SILVER_LABELS; do
      emit "$ROOT/3_shared_encoder/4_silver_encoder/${label}.sh" "WEIGHT_SOURCE=$weight"
    done
  done
fi

if has_stage diagnosis; then
  for weight in $WEIGHT_SOURCES; do
    for mode in $DX_MODES; do
      for strategy in $DX_STRATEGIES; do
        if [[ "$mode" == "multi_task" ]]; then
          [[ "$strategy" == "global" ]] || continue
          emit "$ROOT/4_diagnosis/multi-task/${strategy}.sh" "WEIGHT_SOURCE=$weight"
        else
          for label in $DX_LABELS; do
            local_script="$ROOT/4_diagnosis/single-task/${label}/${strategy}.sh"
            [[ -f "$local_script" ]] || continue
            emit "$local_script" "WEIGHT_SOURCE=$weight"
          done
        fi
      done
    done
  done
fi

if has_stage prognosis; then
  for weight in $WEIGHT_SOURCES; do
    for cohort in $COHORTS; do
      for profile in $EHR_PROFILES; do
        for task in $TASKS; do
          for strategy in $STRATEGIES; do
            if [[ "$TABULAR_ONLY" == *" $strategy "* && "$weight" != "$TABULAR_WEIGHT_SOURCE" ]]; then
              continue
            fi
            emit "$ROOT/5_prognosis/${cohort}/${profile}/${task}/${strategy}.sh" \
              "WEIGHT_SOURCE=$weight"
          done
        done
      done
    done
  done
fi

printf '\n%s run(s) in the selected matrix.\n' "$planned"
if [[ -z "${EXECUTE:-}" ]]; then
  printf 'Nothing was executed. Re-run with EXECUTE=1 (add ACTION=preflight to check first).\n'
elif (( failed )); then
  printf '%s run(s) failed:\n' "$failed"
  printf '  %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi
