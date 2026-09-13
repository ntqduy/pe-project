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
CLINICAL_TABLES = ("ehr_features.csv", "pesi_features.csv")


@dataclass
class DatasetBundle:
    """Whatever a dataset profile actually has on disk, plus what it is missing."""

    profile: str
    dataset_root: Path
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unreadable: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

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
                "path": str((self.dataset_root / _relative(name)).resolve()),
            }
            for name, rows in sorted(self.tables.items())
        ]
        inventory += [
            {"table": name, "status": "missing", "rows": 0, "columns": 0,
             "path": str((self.dataset_root / _relative(name)).resolve())}
            for name in sorted(self.missing)
        ]
        inventory += [
            {"table": name, "status": f"unreadable: {reason}", "rows": 0, "columns": 0,
             "path": str((self.dataset_root / _relative(name)).resolve())}
            for name, reason in sorted(self.unreadable.items())
        ]
        return inventory


def _relative(name: str) -> str:
    return f"clinical/{name}" if name in CLINICAL_TABLES else f"manifests/{name}"


def load_dataset(dataset_root: Path, profile: str) -> DatasetBundle:
    """Read every known table, recording what is absent instead of failing on it.

    A profile that has not been built yet is a legitimate EDA input: the report then says
    exactly which stage-0 artifacts are missing rather than raising.
    """
    bundle = DatasetBundle(profile=profile, dataset_root=dataset_root)
    for name in (*KNOWN_MANIFESTS, *CLINICAL_TABLES):
        path = dataset_root / _relative(name)
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
