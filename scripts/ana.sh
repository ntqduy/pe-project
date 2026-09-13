#!/usr/bin/env bash
# Exploratory data analysis for one built dataset profile.
#
#   bash scripts/ana.sh                              # PROFILE=test_500_sample
#   PROFILE=full_inspect bash scripts/ana.sh
#   PROFILE=smoke_30 RULE_SAMPLE=0 bash scripts/ana.sh   # 0 = scan every report
#   NO_FIGURES=1 bash scripts/ana.sh                 # numbers only, no PNG
#
# Environment (all optional):
#   PROFILE       smoke_30 | test_500_sample | full_inspect   default: test_500_sample
#   DATASET_ROOT  override the resolved dataset root
#   OUTPUT_ROOT   override the resolved output root
#   RULE_SAMPLE   reports scanned for rule coverage           default: 2000
#   NO_FIGURES=1  skip PNG rendering
#
# Output: <output_root>/EDA/<profile>/  — EDA_REPORT.md, summary.json, *.csv,
# figures/*.png, logs/run.log. Reads derived artifacts only; writes nothing that a
# training stage consumes, so it is safe to re-run at any time.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PE_CLOUD_ROOT:-}" && -f "$PROJECT_ROOT/scripts/use_gcs_storage.sh" ]]; then
  # shellcheck disable=SC1091 -- intentional repository-local environment helper
  source "$PROJECT_ROOT/scripts/use_gcs_storage.sh"
fi
PYTHON="${PYTHON:-python}"

declare -a args=(--profile "${PROFILE:-test_500_sample}" --rule-sample "${RULE_SAMPLE:-2000}")
[[ -n "${DATASET_ROOT:-}" ]] && args+=(--dataset-root "$DATASET_ROOT")
[[ -n "${OUTPUT_ROOT:-}" ]] && args+=(--output-root "$OUTPUT_ROOT")
[[ -n "${NO_FIGURES:-}" ]] && args+=(--no-figures)

printf '==> EDA  [profile=%s]\n' "${PROFILE:-test_500_sample}"
exec "$PYTHON" "$PROJECT_ROOT/analysis/run_eda.py" "${args[@]}"
