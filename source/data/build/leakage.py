"""Official-split preservation and patient-leakage guards.

Ported from /mnt/pe_study `verify_inspect_subset.py` (official split preservation and
patient linkage) and `build_seen_patient_registry.py` (governance exclusion registry).

These checks are cheap and run on every dataset build, for both profiles. A leak found
after training has already cost the experiment; a leak found here costs a rerun of a
metadata pass.
"""
from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sources import StudyRecord, normalize_split


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class SplitAudit:
    patients: int
    studies: int
    split_patients: dict[str, int]
    split_studies: dict[str, int]
    cross_split_patients: tuple[str, ...]
    duplicate_studies: tuple[str, ...]
    split_reassignments: tuple[str, ...]
    unknown_studies: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "ok": self.ok,
            "patients": self.patients,
            "studies": self.studies,
            "split_patients": self.split_patients,
            "split_studies": self.split_studies,
            "cross_split_patients": list(self.cross_split_patients),
            "duplicate_studies": list(self.duplicate_studies),
            "split_reassignments": list(self.split_reassignments),
            "unknown_studies": list(self.unknown_studies),
            "errors": list(self.errors),
        }
        return payload


def load_excluded_patients(paths: Iterable[str | Path]) -> set[str]:
    """Read patient ids from governance registries, ignoring every other field.

    Accepts CSV or TSV with either ``person_id`` (INSPECT) or ``patient_id`` (project).
    """
    excluded: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise LeakageError(f"patient exclusion registry not found: {path}")
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
        if not rows:
            continue
        column = "person_id" if "person_id" in rows[0] else "patient_id"
        if column not in rows[0]:
            raise LeakageError(f"{path} must contain a person_id or patient_id column")
        excluded.update(str(row.get(column) or "").strip() for row in rows)
    excluded.discard("")
    return excluded


def audit_split_integrity(
    records: Sequence[StudyRecord],
    official_splits: Mapping[str, str] | None = None,
    *,
    require_all_splits: bool = True,
) -> SplitAudit:
    """Check patient leakage, duplicate studies and preservation of the official split.

    ``official_splits`` maps ``study_id -> official split``; when given, every study must
    keep the split the release assigned it. That is what makes a sampled cohort
    comparable with the full cohort.
    """
    patient_splits: dict[str, set[str]] = {}
    split_studies: Counter[str] = Counter()
    split_patients: dict[str, set[str]] = {}
    study_ids: list[str] = []
    reassignments: list[str] = []
    unknown: list[str] = []
    for record in records:
        patient_splits.setdefault(record.patient_id, set()).add(record.split)
        split_patients.setdefault(record.split, set()).add(record.patient_id)
        split_studies[record.split] += 1
        study_ids.append(record.study_id)
        if official_splits is not None:
            expected = official_splits.get(record.study_id)
            if expected is None:
                unknown.append(record.study_id)
            elif normalize_split(expected) != record.split:
                reassignments.append(
                    f"{record.study_id}: cohort={record.split} official={normalize_split(expected)}"
                )
    cross = tuple(sorted(key for key, value in patient_splits.items() if len(value) > 1))
    duplicates = tuple(sorted(key for key, count in Counter(study_ids).items() if count > 1))

    errors: list[str] = []
    if cross:
        errors.append(f"{len(cross)} patient(s) appear in more than one split")
    if duplicates:
        errors.append(f"{len(duplicates)} duplicate study id(s)")
    if reassignments:
        errors.append(f"{len(reassignments)} study/studies changed split relative to the release")
    if unknown:
        errors.append(f"{len(unknown)} study/studies are absent from the official split table")
    if require_all_splits:
        missing = sorted({"train", "validation", "test"} - set(split_studies))
        if missing:
            errors.append("cohort has no rows for split(s): " + ", ".join(missing))
    return SplitAudit(
        patients=len(patient_splits),
        studies=len(set(study_ids)),
        split_patients={key: len(value) for key, value in sorted(split_patients.items())},
        split_studies=dict(sorted(split_studies.items())),
        cross_split_patients=cross[:50],
        duplicate_studies=duplicates[:50],
        split_reassignments=tuple(reassignments[:50]),
        unknown_studies=tuple(sorted(set(unknown))[:50]),
        errors=tuple(errors),
    )


def require_no_leakage(audit: SplitAudit) -> SplitAudit:
    if not audit.ok:
        raise LeakageError("; ".join(audit.errors))
    return audit
