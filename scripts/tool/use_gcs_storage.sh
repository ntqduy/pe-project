#!/usr/bin/env bash
# Default storage roots for this project; values already exported take precedence.
#
#   source scripts/tool/use_gcs_storage.sh            # current shell only (all run_*.sh do this)
#   bash scripts/tool/use_gcs_storage.sh --install    # add it to ~/.bashrc once: every new
#                                                # terminal then has the variables, so
#                                                # `python run.py ...` works without exports
#   bash scripts/tool/use_gcs_storage.sh --show       # print the resolved values and checks
#   bash scripts/tool/use_gcs_storage.sh --uninstall  # remove the ~/.bashrc line again
#
# Safe to source from ~/.bashrc: it never exits the shell and only prints warnings.

# Same defaults as configs/paths.yaml: everything the pipeline writes goes under
# /mnt/pe-project/outputs (derived/ and pe-project/outputs/, the layout of
# gs://pe-study/pe-storage), and the raw release is read from /mnt/Stanford_INSPECT_dataset.
# Until 2026-09-27 the bucket was mounted at /mnt/pe-storage and used as the root; a shell
# started before then still exports those values, so they are dropped here.
if [[ "${PE_CLOUD_ROOT:-}" == "/mnt/pe-storage" ]]; then
  unset PE_CLOUD_ROOT PE_DERIVED_ROOT PE_CLOUD_PROJECT_ROOT
fi
if [[ "${PE_CLOUD_PROJECT_ROOT:-}" == "/mnt/pe-project" ]]; then
  unset PE_CLOUD_PROJECT_ROOT
fi
export PE_CLOUD_ROOT="${PE_CLOUD_ROOT:-/mnt/pe-project/outputs}"
export PE_RAW_INSPECT_ROOT="${PE_RAW_INSPECT_ROOT:-/mnt/Stanford_INSPECT_dataset}"
export PE_DERIVED_ROOT="${PE_DERIVED_ROOT:-${PE_CLOUD_ROOT}/derived}"
export PE_CLOUD_PROJECT_ROOT="${PE_CLOUD_PROJECT_ROOT:-${PE_CLOUD_ROOT}/pe-project}"

_pe_storage_check() {
  if [[ ! -d "$PE_DERIVED_ROOT/datasets" ]]; then
    printf 'warning: no dataset profiles under %s/datasets\n' "$PE_DERIVED_ROOT" >&2
  fi
  if [[ ! -d "$PE_RAW_INSPECT_ROOT/CT/full" ]]; then
    printf 'warning: raw INSPECT release not found at %s/CT/full; mount it with\n' "$PE_RAW_INSPECT_ROOT" >&2
    printf '         gcsfuse --implicit-dirs --only-dir Stanford_INSPECT_dataset pe-study %s\n' "$PE_RAW_INSPECT_ROOT" >&2
  fi
}

# Warn once per shell, not on every script that sources this file.
if [[ -z "${_PE_STORAGE_CHECKED:-}" ]]; then
  export _PE_STORAGE_CHECKED=1
  _pe_storage_check
fi

# Executed (not sourced): handle --install / --uninstall / --show.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  _pe_helper="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  _pe_line="[ -f \"$_pe_helper\" ] && source \"$_pe_helper\"  # pe-project storage roots"
  _pe_rc="${HOME}/.bashrc"
  case "${1:---show}" in
    --install)
      if grep -Fq "# pe-project storage roots" "$_pe_rc" 2>/dev/null; then
        printf 'already installed in %s\n' "$_pe_rc"
      else
        printf '\n%s\n' "$_pe_line" >> "$_pe_rc"
        printf 'added to %s; open a new terminal or run: source %s\n' "$_pe_rc" "$_pe_rc"
      fi
      ;;
    --uninstall)
      if [[ -f "$_pe_rc" ]]; then
        sed -i '/# pe-project storage roots$/d' "$_pe_rc"
        printf 'removed from %s\n' "$_pe_rc"
      fi
      ;;
    --show)
      _pe_storage_check
      printf 'PE_CLOUD_ROOT=%s\nPE_CLOUD_PROJECT_ROOT=%s\nPE_RAW_INSPECT_ROOT=%s\nPE_DERIVED_ROOT=%s\n' \
        "$PE_CLOUD_ROOT" "$PE_CLOUD_PROJECT_ROOT" "$PE_RAW_INSPECT_ROOT" "$PE_DERIVED_ROOT"
      ;;
    *)
      printf 'usage: bash %s [--install|--uninstall|--show]\n' "$0" >&2
      exit 2
      ;;
  esac
fi
