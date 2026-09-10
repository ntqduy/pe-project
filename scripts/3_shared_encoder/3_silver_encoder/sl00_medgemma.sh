#!/usr/bin/env bash
# Deprecated compatibility path. Silver encoder now follows RSPECT under 4_silver_encoder.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/../4_silver_encoder/sl00_medgemma.sh" "$@"
