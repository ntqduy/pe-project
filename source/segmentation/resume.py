"""Resume guard for the data-generation stages (segmentation, ROI construction).

Per-study state files are reused on resume. They are only valid for the settings that
produced them, so every run directory records those settings in ``resume_settings.json``
and a later invocation with different output-affecting settings fails loudly instead of
silently mixing cached studies built one way with new studies built another way.

The run-level comparison and the per-study fingerprint use one canonical form
(:func:`canonical_settings`): an integral float equals the integer (``5.0`` is ``5``), while
a bool stays distinct from an int (``true`` is not ``1``). Settings that pass the run guard
therefore keep every cached study valid, and a change that would alter the fingerprint is
reported up front instead of silently recomputing every study.

Runs recorded before canonicalization stay resumable: per-study states may carry the
legacy fingerprint of the verified settings (:class:`SettingsFingerprint`), and a nested
legacy fingerprint inside recorded settings (the ROI stage records its upstream
segmentation run's) is compared through the canonical fingerprint of the same settings.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

from source.engine.experiment import atomic_write_json

RESUME_SETTINGS_FILE = "resume_settings.json"

# legacy fingerprint -> canonical fingerprint of the same settings, filled by
# settings_fingerprint(); lets a recorded legacy fingerprint compare equal to the canonical
# fingerprint computed now for identical settings.
_LEGACY_EQUIVALENTS: dict[str, str] = {}


class ResumeSettingsMismatch(RuntimeError):
    pass


class SettingsFingerprint(str):
    """The canonical fingerprint, plus the legacy fingerprints that are equally valid.

    Per-study states written before canonicalization carry a fingerprint of the
    non-canonical JSON text. Once the run guard has established that the recorded settings
    are canonically identical to the current ones, those legacy fingerprints identify the
    same settings, so their cached studies are reused instead of all being recomputed.
    New states always record the canonical value (``str(fingerprint)``).
    """

    accepted: frozenset[str]

    def __new__(cls, value: str, legacy: Iterable[str | None] = ()) -> SettingsFingerprint:
        instance = super().__new__(cls, value)
        instance.accepted = frozenset({str(value), *(item for item in legacy if item)})
        return instance


def fingerprint_matches(recorded: Any, expected: str | None) -> bool:
    """Whether a state's recorded fingerprint was produced by the expected settings."""
    if expected is None or recorded is None:
        return recorded == expected
    accepted = getattr(expected, "accepted", None) or frozenset({str(expected)})
    return str(recorded) in accepted


def normalized_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """JSON round trip, so tuples/paths compare equal to their stored list/string form."""
    return json.loads(json.dumps(dict(settings), sort_keys=True, default=str))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    # bool is an int subclass but stays a bool: true and 1 are different settings.
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def canonical_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Normalized settings with every integral float written as an integer."""
    return _canonical_value(normalized_settings(settings))


def _canonical_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def settings_fingerprint(settings: Mapping[str, Any]) -> str:
    payload = _canonical_text(canonical_settings(settings))
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    _LEGACY_EQUIVALENTS[legacy_settings_fingerprint(settings)] = fingerprint
    return fingerprint


def legacy_settings_fingerprint(settings: Mapping[str, Any]) -> str:
    """The pre-canonicalization fingerprint (``5`` and ``5.0`` hashed differently)."""
    payload = json.dumps(normalized_settings(settings), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _upgrade_legacy_fingerprints(value: Any) -> Any:
    """Replace nested legacy fingerprints by the canonical fingerprint of the same settings."""
    if isinstance(value, Mapping):
        return {key: _upgrade_legacy_fingerprints(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_upgrade_legacy_fingerprints(item) for item in value]
    if isinstance(value, str):
        return _LEGACY_EQUIVALENTS.get(value, value)
    return value


def without_keys(mapping: Mapping[str, Any] | None, ignored: Collection[str]) -> dict[str, Any]:
    return {str(key): value for key, value in dict(mapping or {}).items() if str(key) not in ignored}


def _differences(previous: Any, current: Any, prefix: str = "") -> list[str]:
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        changed: list[str] = []
        for key in sorted(set(previous) | set(current), key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous or key not in current:
                changed.append(name)
            else:
                changed.extend(_differences(previous[key], current[key], name))
        return changed
    # Canonical JSON text, not Python equality: True == 1 in Python, but the per-study
    # fingerprint tells them apart.
    same = _canonical_text(previous) == _canonical_text(current)
    return [] if same else [prefix or "<settings>"]


def _previous_settings(
    run_dir: Path,
    from_snapshot: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    record = run_dir / RESUME_SETTINGS_FILE
    if record.is_file():
        payload = json.loads(record.read_text(encoding="utf-8"))
        settings = payload.get("settings") if isinstance(payload, Mapping) else None
        if not isinstance(settings, Mapping):
            raise ResumeSettingsMismatch(f"unreadable resume settings record: {record}")
        return normalized_settings(settings)
    snapshot_path = run_dir / "resolved_config.yaml"
    # Runs written before resume_settings.json existed: the last config snapshot is the
    # best record of the settings that produced their cached studies.
    if from_snapshot is None or not snapshot_path.is_file():
        return None
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to compare a resumed run's config snapshot") from exc
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, Mapping):
        return None
    return normalized_settings(from_snapshot(snapshot))


def _verified_previous(
    run_dir: Path,
    settings: Mapping[str, Any],
    *,
    stage: str,
    from_snapshot: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """The run's recorded settings (as recorded) when they match ``settings``; else raise."""
    if not run_dir.is_dir():
        return None
    previous = _previous_settings(run_dir, from_snapshot)
    if previous is None:
        return None
    current = canonical_settings(settings)
    previous_canonical = _upgrade_legacy_fingerprints(canonical_settings(previous))
    if _canonical_text(previous_canonical) == _canonical_text(current):
        return previous
    changed = _differences(previous_canonical, current)
    raise ResumeSettingsMismatch(
        f"{stage} run {run_dir} was built with different output-affecting settings "
        f"(changed: {', '.join(changed)}). Resuming would silently reuse its cached studies. "
        "Re-run with --overwrite to rebuild the run, restore the original settings, or use "
        "a different experiment id."
    )


def verify_resume_settings(
    run_dir: Path,
    settings: Mapping[str, Any],
    *,
    stage: str,
    from_snapshot: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> None:
    """Raise when an existing run directory was produced with different settings.

    Call it before the run's config snapshot is rewritten: ``from_snapshot`` reads the
    settings out of that snapshot for runs that predate ``resume_settings.json``.
    """
    _verified_previous(run_dir, settings, stage=stage, from_snapshot=from_snapshot)


def _recorded_legacy_fingerprints(run_dir: Path) -> list[str]:
    record = run_dir / RESUME_SETTINGS_FILE
    if not record.is_file():
        return []
    payload = json.loads(record.read_text(encoding="utf-8"))
    values = payload.get("legacy_fingerprints") if isinstance(payload, Mapping) else None
    return [str(value) for value in values] if isinstance(values, list) else []


def record_resume_settings(
    run_dir: Path, settings: Mapping[str, Any], legacy_fingerprints: Iterable[str] = ()
) -> None:
    payload: dict[str, Any] = {
        "fingerprint": settings_fingerprint(settings),
        "settings": normalized_settings(settings),
    }
    legacy = sorted(set(legacy_fingerprints) - {payload["fingerprint"]})
    if legacy:
        # Fingerprints of canonically identical settings that pre-canonicalization states
        # of this run carry; kept so a second resume still accepts them.
        payload["legacy_fingerprints"] = legacy
    atomic_write_json(run_dir / RESUME_SETTINGS_FILE, payload)


def guard_resume_settings(
    run_dir: Path,
    settings: Mapping[str, Any],
    *,
    stage: str,
) -> SettingsFingerprint:
    """Verify against the run's record (if any), then record; returns the fingerprint.

    The result is the canonical fingerprint (a ``str``) that also accepts the legacy
    fingerprints of the current and of the verified recorded settings, so per-study states
    written before canonicalization stay valid (see :func:`fingerprint_matches`).
    """
    previous = _verified_previous(run_dir, settings, stage=stage, from_snapshot=None)
    carried: list[str] = []
    if previous is not None:
        carried = [legacy_settings_fingerprint(previous), *_recorded_legacy_fingerprints(run_dir)]
    record_resume_settings(run_dir, settings, carried)
    return SettingsFingerprint(
        settings_fingerprint(settings), [legacy_settings_fingerprint(settings), *carried]
    )
