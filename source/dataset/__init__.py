"""Dataset profiles: the project's own definition of which patients it trains on.

Exactly two profiles are active:

    test_500_sample   the clean/filtered cohort, then 500 patients sampled patient-level
    full_inspect      the same clean/filtered cohort, no sampling at all

They inherit the identical eligibility, integrity, adjudication, manifest and
preprocessing blocks from ``profiles/_common.yaml`` and differ only in ``sampling``.
`assert_shared_preprocessing()` enforces that at import time of any consumer, so the
two profiles cannot silently drift apart and make their results incomparable.

A profile is data, not code: `source/data_preprocessing/pipeline.build_dataset` is the
single implementation both of them run.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from source.utils.config import ConfigError, deep_merge

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
ACTIVE_PROFILES = ("test_500_sample", "full_inspect")


class DatasetProfileError(ConfigError):
    pass


def available_profiles() -> tuple[str, ...]:
    """Every profile file on disk, active or not."""
    return tuple(
        sorted(path.stem for path in PROFILE_DIR.glob("*.yaml") if not path.stem.startswith("_"))
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise DatasetProfileError("PyYAML is required to read dataset profiles") from exc
    if not path.is_file():
        raise DatasetProfileError(f"dataset profile not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise DatasetProfileError(f"dataset profile must be a mapping: {path}")
    return payload


def _resolve(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        raise DatasetProfileError(f"cyclic dataset profile inheritance: {resolved}")
    payload = _read(resolved)
    bases = payload.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        candidate = Path(str(base))
        if not candidate.is_absolute():
            candidate = resolved.parent / candidate
        merged = deep_merge(merged, _resolve(candidate, (*stack, resolved)))
    return deep_merge(merged, payload)


def load_profile(name: str) -> dict[str, Any]:
    """Resolve one dataset profile, inheritance included."""
    requested = str(name or "").strip()
    if not requested:
        raise DatasetProfileError("a dataset profile name is required")
    if requested not in available_profiles():
        raise DatasetProfileError(
            f"unknown dataset profile {requested!r}; available={available_profiles()}"
        )
    profile = _resolve(PROFILE_DIR / f"{requested}.yaml")
    declared = str((profile.get("profile") or {}).get("name") or "")
    if declared != requested:
        raise DatasetProfileError(
            f"dataset profile {requested!r} declares profile.name={declared!r}"
        )
    return profile


def require_active_profile(name: str) -> dict[str, Any]:
    """Load a profile, refusing anything outside the two active profiles."""
    requested = str(name or "").strip()
    if requested not in ACTIVE_PROFILES:
        raise DatasetProfileError(
            f"dataset profile {requested!r} is not active; use one of {ACTIVE_PROFILES}"
        )
    return load_profile(requested)


def preprocessing_fingerprint(name: str) -> str:
    from source.data_preprocessing.volumes import PreprocessingSpec

    return PreprocessingSpec.from_mapping(load_profile(name).get("preprocessing")).fingerprint()


def assert_shared_preprocessing(names: Mapping[str, Any] | tuple[str, ...] = ACTIVE_PROFILES) -> str:
    """Fail loudly if the active profiles do not share one preprocessing implementation."""
    fingerprints = {name: preprocessing_fingerprint(name) for name in tuple(names)}
    unique = set(fingerprints.values())
    if len(unique) > 1:
        raise DatasetProfileError(
            "active dataset profiles must share one preprocessing contract, but they differ: "
            + ", ".join(f"{key}={value[:12]}" for key, value in sorted(fingerprints.items()))
        )
    return unique.pop()


__all__ = [
    "ACTIVE_PROFILES",
    "DatasetProfileError",
    "assert_shared_preprocessing",
    "available_profiles",
    "load_profile",
    "preprocessing_fingerprint",
    "require_active_profile",
]
