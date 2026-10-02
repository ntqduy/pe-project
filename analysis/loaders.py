"""Locate and load the derived artifacts an EDA run reads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from source.data.manifests import ManifestError, read_rows

# Every table the EDA knows how to describe. Missing ones are reported, never invented.
KNOWN_MANIFESTS = (
    "ctpa.csv",
    "diagnosis.csv",
    "reports.csv",
    "paired_reports.csv",
    "prognosis.csv",
    "prognosis_all_patient.csv",
    "prognosis_pe_positive.csv",
    "prognosis_cohort_membership.csv",
)
# Written by source/data/build/pipeline.py into <derived>/cache/<profile>/clinical/
# (ProjectPaths.dataset_cache_root), not into the dataset folder; see clinical_directories.
CLINICAL_TABLES = ("ehr_features.csv", "spesi_features.csv")


@dataclass
class DatasetBundle:
    """Whatever a dataset profile actually has on disk, plus what it is missing."""

    profile: str
    dataset_root: Path
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unreadable: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    # Where each table was read from (or, when missing, where it was expected first).
    locations: dict[str, Path] = field(default_factory=dict)

    def path_of(self, name: str) -> Path:
        return self.locations.get(name) or (self.dataset_root / _relative(name))

    def rows(self, name: str) -> list[dict[str, Any]]:
        return self.tables.get(name, [])

    def has(self, name: str) -> bool:
        return bool(self.tables.get(name))

    def as_inventory(self) -> list[dict[str, Any]]:
        inventory = [
            {
                "table": name,
                "status": "present",
                "rows": len(rows),
                "columns": len(rows[0]) if rows else 0,
                "path": str(self.path_of(name).resolve()),
            }
            for name, rows in sorted(self.tables.items())
        ]
        inventory += [
            {"table": name, "status": "missing", "rows": 0, "columns": 0,
             "path": str(self.path_of(name).resolve())}
            for name in sorted(self.missing)
        ]
        inventory += [
            {"table": name, "status": f"unreadable: {reason}", "rows": 0, "columns": 0,
             "path": str(self.path_of(name).resolve())}
            for name, reason in sorted(self.unreadable.items())
        ]
        return inventory


def _relative(name: str) -> str:
    return f"clinical/{name}" if name in CLINICAL_TABLES else f"manifests/{name}"


def clinical_directories(dataset_root: Path, profile: str, clinical_root: Path | None = None) -> list[Path]:
    """Where the clinical tables may live, most current layout first.

    The pipeline writes them to ``ProjectPaths.dataset_cache_root(profile) / "clinical"``,
    i.e. ``<derived>/cache/<profile>/clinical``. When the caller does not pass that directory
    it is inferred from the standard ``<derived>/datasets/<profile>`` dataset location.
    Profiles built before the cache split kept them in ``<dataset_root>/clinical``.
    """
    candidates: list[Path] = []
    if clinical_root is not None:
        candidates.append(Path(clinical_root))
    if dataset_root.parent.name == "datasets":
        candidates.append(dataset_root.parent.parent / "cache" / profile / "clinical")
    candidates.append(dataset_root / "clinical")
    return list(dict.fromkeys(candidates))


def load_dataset(dataset_root: Path, profile: str, clinical_root: Path | None = None) -> DatasetBundle:
    """Read every known table, recording what is absent instead of failing on it.

    A profile that has not been built yet is a legitimate EDA input: the report then says
    exactly which stage-0 artifacts are missing rather than raising. ``clinical_root`` is the
    profile's clinical cache directory (see clinical_directories).
    """
    bundle = DatasetBundle(profile=profile, dataset_root=dataset_root)
    clinical = clinical_directories(dataset_root, profile, clinical_root)
    for name in (*KNOWN_MANIFESTS, *CLINICAL_TABLES):
        if name in CLINICAL_TABLES:
            candidates = [directory / name for directory in clinical]
        else:
            candidates = [dataset_root / _relative(name)]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        bundle.locations[name] = path
        if not path.is_file():
            bundle.missing.append(name)
            continue
        try:
            bundle.tables[name] = read_rows(path)
        except (ManifestError, OSError, ValueError) as exc:
            bundle.unreadable[name] = f"{type(exc).__name__}: {exc}"
    provenance = dataset_root / "dataset.json"
    if provenance.is_file():
        try:
            import json

            bundle.provenance = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            bundle.provenance = {"error": f"{type(exc).__name__}: {exc}"}
    return bundle


def column_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)
