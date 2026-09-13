"""Read-only access to the raw Stanford INSPECT release.

Ported and generalized from /mnt/pe_study `build_inspect_manifest.py` and
`create_inspect_subset.py`. Nothing in this module writes to the raw tree: the raw
release is an input, never an output.

The release ships date-stamped TSV tables (``splits_20250611.tsv`` and friends). The
stamp is discovered by globbing rather than hardcoded, so a newer release drops in
without a code change.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The official release names the splits train/valid/test; the project manifest schema
# (source/data/manifests.py ALLOWED_SPLITS) names them train/validation/test.
SPLIT_ALIASES = {"train": "train", "valid": "validation", "validation": "validation", "test": "test"}
OFFICIAL_SPLITS = ("train", "validation", "test")

TABLES = {
    "splits": "splits",
    "labels": "labels",
    "study_mapping": "study_mapping",
    "series_metadata": "series_metadata",
    "study_metadata": "study_metadata",
    "impressions": "impressions",
}
CROSSWALK_GLOB = "image_ehr_crosswalk_*.csv"

# Series/study metadata carry one row per DICOM instance; only these fields are
# reused downstream and they are constant within a series.
SERIES_FIELDS = (
    "Manufacturer",
    "num_slices",
    "SliceThickness",
    "PixelSpacing_0",
    "PixelSpacing_1",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
)
STUDY_FIELDS = (
    "StudyDescription",
    "Modality",
    "KVP",
    "ConvolutionKernel",
    "PatientPosition",
    "Rows",
    "Columns",
)


class InspectSourceError(RuntimeError):
    pass


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_split(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return SPLIT_ALIASES.get(raw, raw)


def read_table(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


@dataclass(frozen=True)
class StudyRecord:
    """One CTPA study joined across every official table.

    ``patient_id``/``study_id`` use the project's vocabulary; ``person_id``/``image_id``
    are the INSPECT identifiers they come from, kept so results stay traceable.
    """

    patient_id: str
    study_id: str
    impression_id: str
    split: str
    note_id: str
    procedure_datetime: str
    note_datetime: str
    provenance: str
    report_text: str
    image_path: str
    labels: Mapping[str, str] = field(default_factory=dict)
    series: Mapping[str, str] = field(default_factory=dict)
    study: Mapping[str, str] = field(default_factory=dict)
    has_ehr_crosswalk: bool = False

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "impression_id": self.impression_id,
            "person_id": self.patient_id,
            "image_id": self.study_id,
            "split": self.split,
            "note_id": self.note_id,
            "procedure_datetime": self.procedure_datetime,
            "note_datetime": self.note_datetime,
            "provenance": self.provenance,
            "report_text": self.report_text,
            "image_path": self.image_path,
            "has_ehr_crosswalk": "TRUE" if self.has_ehr_crosswalk else "FALSE",
        }
        row.update({key: value for key, value in self.labels.items()})
        row.update({key: value for key, value in self.series.items()})
        row.update({key: value for key, value in self.study.items()})
        return row


class InspectSource:
    """Join the official INSPECT tables into one study-level record set.

    ``root`` is the release directory that directly contains the TSV tables and the
    ``CTPA/`` volume directory, e.g. ``<raw_inspect>/CT/full`` or ``<raw_inspect>/CT/sample``.
    """

    def __init__(self, root: str | Path, *, volume_suffix: str = ".nii.gz"):
        self.root = Path(root)
        if not self.root.is_dir():
            raise InspectSourceError(f"INSPECT release directory not found: {self.root}")
        self.volume_suffix = volume_suffix
        self.volume_root = self.root / "CTPA"
        self.tables = {name: self._locate(stem) for name, stem in TABLES.items()}

    def _locate(self, stem: str) -> Path:
        matches = sorted(self.root.glob(f"{stem}_*.tsv")) or sorted(self.root.glob(f"{stem}.tsv"))
        if not matches:
            raise InspectSourceError(f"required INSPECT table not found: {self.root}/{stem}_*.tsv")
        return matches[-1]

    def release_stamp(self) -> str:
        match = re.search(r"_(\d{8})\.tsv$", self.tables["splits"].name)
        return match.group(1) if match else "unknown"

    def _crosswalk_keys(self) -> set[tuple[str, str]]:
        candidates = sorted((self.root / "EHR").glob(CROSSWALK_GLOB)) if (self.root / "EHR").is_dir() else []
        candidates += sorted(self.root.glob(CROSSWALK_GLOB))
        if not candidates:
            return set()
        keys: set[tuple[str, str]] = set()
        for row in read_table(candidates[-1], ","):
            person = str(row.get("person_id") or "").strip()
            image = str(row.get("image_id") or "").strip()
            if person and image:
                keys.add((person, image))
        return keys

    @staticmethod
    def _primary_series(rows: Iterable[Mapping[str, str]]) -> dict[str, dict[str, str]]:
        """Pick the diagnostic reconstruction for each image, not just the first row.

        ``series_metadata`` carries one row per *series* in the study: localizers and
        thick scouts sit next to the thin CTPA reconstruction. Reading the first (or the
        last) row -- which is what the original /mnt/pe_study scripts did -- reports the
        localizer's geometry, and the noise-removal rules then discard a fifth of the
        release as "1-slice" studies. The diagnostic series is the one with the most
        slices; ties are broken towards the thinner reconstruction.
        """
        best: dict[str, tuple[float, float, dict[str, str]]] = {}
        for row in rows:
            identifier = str(row.get("image_id") or "").strip()
            if not identifier:
                continue
            try:
                slices = float(str(row.get("num_slices") or "").strip() or 0)
            except ValueError:
                slices = 0.0
            try:
                thickness = float(str(row.get("SliceThickness") or "").strip() or "inf")
            except ValueError:
                thickness = float("inf")
            rank = (slices, -thickness)
            current = best.get(identifier)
            if current is None or rank > (current[0], -current[1]):
                best[identifier] = (
                    slices,
                    thickness,
                    {name: str(row.get(name) or "").strip() for name in SERIES_FIELDS},
                )
        return {identifier: payload for identifier, (_, _, payload) in best.items()}

    @staticmethod
    def _modal_study(rows: Iterable[Mapping[str, str]], fields: Sequence[str]) -> dict[str, dict[str, str]]:
        """Reduce the per-instance study metadata to the most common value per field.

        Like the series table, ``study_metadata`` repeats per acquisition. Taking the most
        frequent non-empty value is stable under row order and never invents a value that
        no row carried.
        """
        counters: dict[str, dict[str, Counter[str]]] = {}
        for row in rows:
            identifier = str(row.get("impression_id") or "").strip()
            if not identifier:
                continue
            per_field = counters.setdefault(identifier, {name: Counter() for name in fields})
            for name in fields:
                value = str(row.get(name) or "").strip()
                if value:
                    per_field[name][value] += 1
        return {
            identifier: {
                name: (counter.most_common(1)[0][0] if counter else "")
                for name, counter in per_field.items()
            }
            for identifier, per_field in counters.items()
        }

    def volume_path(self, study_id: str) -> Path:
        return self.volume_root / f"{study_id}{self.volume_suffix}"

    def records(self) -> list[StudyRecord]:
        splits = {row["impression_id"]: row for row in read_table(self.tables["splits"])}
        labels = {row["impression_id"]: row for row in read_table(self.tables["labels"])}
        impressions = {
            row["impression_id"]: normalize_text(row.get("impressions"))
            for row in read_table(self.tables["impressions"])
        }
        series = self._primary_series(read_table(self.tables["series_metadata"]))
        studies = self._modal_study(read_table(self.tables["study_metadata"]), STUDY_FIELDS)
        crosswalk = self._crosswalk_keys()

        records: list[StudyRecord] = []
        for row in read_table(self.tables["study_mapping"]):
            impression_id = str(row.get("impression_id") or "").strip()
            person_id = str(row.get("person_id") or "").strip()
            image_id = str(row.get("image_id") or "").strip()
            split_row = splits.get(impression_id)
            if split_row is None:
                # No official split assignment: this study is not part of the release cohort.
                continue
            if str(split_row.get("person_id") or "").strip() != person_id:
                raise InspectSourceError(
                    f"person mismatch for impression_id={impression_id}: "
                    f"mapping={person_id} splits={split_row.get('person_id')}"
                )
            label_row = {
                key: str(value or "").strip()
                for key, value in (labels.get(impression_id) or {}).items()
                if key != "impression_id"
            }
            records.append(
                StudyRecord(
                    patient_id=person_id,
                    study_id=image_id,
                    impression_id=impression_id,
                    split=normalize_split(split_row.get("split")),
                    note_id=str(row.get("note_id") or "").strip(),
                    procedure_datetime=str(row.get("procedure_DATETIME") or "").strip(),
                    note_datetime=str(row.get("note_DATETIME") or "").strip(),
                    provenance=str(row.get("provenance") or "").strip(),
                    report_text=impressions.get(impression_id, ""),
                    image_path=str(self.volume_path(image_id)) if image_id else "",
                    labels=label_row,
                    series=series.get(image_id, {name: "" for name in SERIES_FIELDS}),
                    study=studies.get(impression_id, {name: "" for name in STUDY_FIELDS}),
                    has_ehr_crosswalk=(person_id, image_id) in crosswalk,
                )
            )
        if not records:
            raise InspectSourceError(f"no INSPECT studies could be joined under {self.root}")
        return records

    def label_columns(self) -> tuple[str, ...]:
        rows = read_table(self.tables["labels"])
        if not rows:
            return ()
        return tuple(name for name in rows[0] if name != "impression_id")

    def provenance(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "release_stamp": self.release_stamp(),
            "tables": {name: str(path) for name, path in sorted(self.tables.items())},
            "volume_root": str(self.volume_root),
            "volume_suffix": self.volume_suffix,
        }
