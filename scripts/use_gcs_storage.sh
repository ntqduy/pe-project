#!/usr/bin/env bash
# Default storage roots for project scripts; existing exports take precedence.

export PE_CLOUD_ROOT="${PE_CLOUD_ROOT:-/mnt/pe-storage}"
export PE_RAW_INSPECT_ROOT="${PE_RAW_INSPECT_ROOT:-/mnt/Stanford_INSPECT_dataset}"
export PE_DERIVED_ROOT="${PE_DERIVED_ROOT:-${PE_CLOUD_ROOT}/derived}"
