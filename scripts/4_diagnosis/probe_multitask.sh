#!/usr/bin/env bash
# Frozen-encoder probe over the three native diagnosis labels (pe_positive, pe_acute,
# pe_subsegmental). Single-task counterpart: probe.sh.
#
# Thin wrapper: resolves probe.diag.multitask through run.py. No scientific logic lives here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"
pe_run probe.diag.multitask "$@"
