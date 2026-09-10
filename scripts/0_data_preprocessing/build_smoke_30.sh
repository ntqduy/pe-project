#!/usr/bin/env bash
# Build the small, deterministic all-split technical smoke cohort.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

# Unlike every other generation wrapper, this profile is intrinsically bounded to thirty
# patients.  Make the safe technical rehearsal one command; callers may still provide a
# narrower MAX_CASES or PATIENT_ID scope explicitly when debugging.
if [[ -z "${ALLOW_ALL:-}" && -z "${MAX_CASES:-}" && -z "${MAX_REPORTS:-}" && -z "${PATIENT_ID:-}" ]]; then
  export ALLOW_ALL=1
fi
pe_run --scoped data.dataset.smoke_30 "$@"
