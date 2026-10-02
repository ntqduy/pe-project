#!/usr/bin/env bash
# Boolean environment flags shared by every scripts/**/*.sh (source it, do not run it).
#
#   is_true OVERWRITE      # on only when OVERWRITE is 1/true/yes/on; unset or empty = off
#   is_false EPOCH_AUC     # default-on flags: succeeds only for 0/false/no/off; unset = on
#
# Matching ignores case. Any other non-empty value (OVERWRITE=maybe) stops the script with
# exit 2 instead of being silently read as "off". The functions take the variable NAME so
# the error can say which flag was wrong; call them in the main shell, never inside $(...),
# or the exit only leaves the subshell.

# _flag_value NAME -> 1 | 0 | invalid (stdout), "" when unset or empty.
_flag_value() {
  local value="${!1:-}"
  [[ -z "$value" ]] && return 0
  case "$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) printf '1' ;;
    0|false|no|off) printf '0' ;;
    *) printf 'invalid' ;;
  esac
}

_flag_check() {
  if [[ "$2" == "invalid" ]]; then
    printf 'error: %s=%s is not a boolean (use 1/0, true/false, yes/no or on/off)\n' "$1" "${!1}" >&2
    exit 2
  fi
}

is_true() {
  local parsed
  parsed="$(_flag_value "$1")"
  _flag_check "$1" "$parsed"
  [[ "$parsed" == "1" ]]
}

is_false() {
  local parsed
  parsed="$(_flag_value "$1")"
  _flag_check "$1" "$parsed"
  [[ "$parsed" == "0" ]]
}
