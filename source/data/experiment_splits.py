"""Patient-level training-fraction manifests of the official INSPECT split (exp02).

The official split is the only split: validation and test are never touched, only the
official TRAIN patients are subsampled. Every fraction is derived from one persisted,
patient-level assignment per (task, label, seed), so every model sees exactly the same patients:

    assignment.csv   patient_id, split (official), label, fraction_rank
      fraction_rank  a per-label uniform rank in [0, 1) over the official train patients
                     (patients shuffled per label with ``seed``); empty for validation / test

    frac  p        : keep train patients with fraction_rank < p (validation, test untouched)

Because the rank is per label, every fraction is stratified; because it is a fixed rank,
fractions are nested (25% c 50% c 75% c 100%); a patient keeps all of its studies. Output
layout, under the dataset root:

    manifests/experiments/<task>_<label>_s<seed>/assignment.csv, summary.json
    manifests/experiments/<task>_<label>_s<seed>/frac{PPP}/<source manifest path>
    manifests/experiments/<task>_<label>_s<seed>/frac{PPP}/<source manifest dir>/train_patients.csv

``frac{PPP}`` is the whole training percent, zero-padded (frac025, frac100); a fractional
percent keeps its decimals after a ``p`` (12.5% -> frac012p5), so distinct fractions never
share a folder. ``train_patients.csv`` sits next to the derived manifest it was taken from
(e.g. ``frac025/manifests/ct_fm/``), because two source manifests can differ in patients.

Image paths in a manifest are relative to the dataset root, so the copies read the same files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .manifests import ManifestError, read_rows

ASSIGNMENT_FIELDS = ("patient_id", "split", "label", "fraction_rank")


def split_name(task: str, label: str, seed: int) -> str:
    return f"{task}_{label}_s{int(seed)}"


def fraction_tag(fraction: float) -> str:
    # Rounded to 1e-6 % only to absorb float noise (0.1 * 100 = 10.000000000000002).
    percent = round(float(fraction) * 100, 6)
    if not 0.0 < percent <= 100.0:
        raise ValueError(f"training fraction must be in (0, 1], got {fraction!r}")
    whole, _, decimals = f"{percent:.6f}".rstrip("0").partition(".")
    return f"frac{int(whole):03d}" + (f"p{decimals}" if decimals else "")


def parse_fraction(value: str | float) -> float:
    """Training fraction in (0, 1] from a float fraction or a CLI string.

    A float is already a fraction. A string is a percent when it ends in ``%`` or is
    greater than 1 ("25", "12.5", "100"), and a fraction when below 1 ("0.25"). A bare
    "1" could mean 1% or 100%, so it is rejected: write "100" or "1%".
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        text = str(value).strip()
        percent = text.endswith("%")
        number = float(text.removesuffix("%"))
        if not percent and number == 1.0:
            raise ValueError(f"training fraction {value!r} is ambiguous; write 100 (all) or 1% (one percent)")
        number = number / 100.0 if percent or number > 1.0 else number
    if not 0.0 < number <= 1.0:
        raise ValueError(f"training fraction must be in (0, 1] or (0%, 100%], got {value!r}")
    return number


def _label(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def build_assignment(rows: Iterable[Mapping[str, Any]], label: str, seed: int) -> list[dict[str, Any]]:
    """One row per patient; ``label`` = max observed study label (``-1`` when none)."""
    split_of: dict[str, str] = {}
    label_of: dict[str, float] = {}
    for row in rows:
        patient = str(row.get("patient_id") or "").strip()
        split = str(row.get("split") or "").strip()
        if not patient:
            raise ManifestError("manifest row without patient_id")
        if split_of.setdefault(patient, split) != split:
            raise ManifestError(f"patient {patient} appears in two official splits")
        value = _label(row.get(label))
        if value is not None:
            label_of[patient] = max(label_of.get(patient, value), value)
    rng = random.Random(int(seed))
    by_label: dict[float, list[str]] = defaultdict(list)
    for patient in sorted(split_of):
        if split_of[patient] == "train":
            by_label[label_of.get(patient, -1.0)].append(patient)
    rank_of: dict[str, float] = {}
    for group_label in sorted(by_label):
        ranked = list(by_label[group_label])
        rng.shuffle(ranked)
        for index, patient in enumerate(ranked):
            rank_of[patient] = index / len(ranked)
    return [
        {
            "patient_id": patient,
            "split": split_of[patient],
            "label": label_of.get(patient, -1.0),
            "fraction_rank": "" if patient not in rank_of else f"{rank_of[patient]:.8f}",
        }
        for patient in sorted(split_of)
    ]


def _keeps(entry: Mapping[str, Any], fraction: float) -> bool:
    """Whether a patient stays in the fraction-``fraction`` manifest (validation / test always do)."""
    if str(entry["split"]) != "train":
        return True
    return float(entry["fraction_rank"]) < float(fraction)


class ExperimentSplits:
    """Persisted assignment + on-demand training-fraction manifests under one dataset root."""

    def __init__(self, dataset_root: Path, *, task: str, label: str, base_manifest: str, seed: int = 42):
        self.dataset_root = Path(dataset_root)
        self.task, self.label, self.seed = task, label, int(seed)
        self.base_manifest = base_manifest
        self.name = split_name(task, label, seed)
        self.root = self.dataset_root / "manifests" / "experiments" / self.name

    def _base_fingerprint(self) -> str:
        """Patient -> official split -> label of the base manifest (what the assignment encodes)."""
        rows = read_rows(self.dataset_root / self.base_manifest)
        items = sorted({(str(r.get("patient_id")), str(r.get("split")), str(r.get(self.label))) for r in rows})
        return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()

    def assignment(self) -> dict[str, dict[str, Any]]:
        path = self.root / "assignment.csv"
        if path.is_file():
            summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8")) if (self.root / "summary.json").is_file() else {}
            recorded = summary.get("base_fingerprint")
            if recorded and recorded != self._base_fingerprint():
                raise ManifestError(
                    f"{self.base_manifest} changed since {path} was written (dataset rebuilt?); "
                    f"delete {self.root} to re-derive the training fractions for the new cohort"
                )
        if not path.is_file():
            rows = read_rows(self.dataset_root / self.base_manifest)
            table = build_assignment(rows, self.label, self.seed)
            _atomic_csv(path, table, ASSIGNMENT_FIELDS)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            counts = Counter((row["split"], row["label"]) for row in table)
            _atomic_json(self.root / "summary.json", {
                "name": self.name, "base_manifest": self.base_manifest, "label": self.label,
                "base_fingerprint": self._base_fingerprint(),
                "seed": self.seed, "patients": len(table), "assignment_sha256": digest,
                "patients_by_split_label": {"|".join(map(str, key)): value for key, value in sorted(counts.items())},
                "rules": __doc__,
            })
        return {str(row["patient_id"]): row for row in read_rows(path)}

    def manifest_path(self, source_manifest: str, fraction: float) -> Path:
        return self.root / fraction_tag(fraction) / source_manifest

    def relative_manifest(self, source_manifest: str, fraction: float) -> str:
        return self.manifest_path(source_manifest, fraction).relative_to(self.dataset_root).as_posix()

    def materialize(self, source_manifest: str, fraction: float) -> dict[str, Any]:
        fraction = float(fraction)
        destination = self.manifest_path(source_manifest, fraction)
        assignment = self.assignment()
        source_path = self.dataset_root / source_manifest
        rows = read_rows(source_path)
        if not rows:
            raise ManifestError(f"empty manifest: {source_path}")
        unknown = sorted({str(row["patient_id"]) for row in rows} - set(assignment))
        if unknown:
            raise ManifestError(
                f"{len(unknown)} patients of {source_manifest} are not in the {self.name} assignment "
                f"(built from {self.base_manifest}); first: {unknown[:5]}"
            )
        kept: list[dict[str, Any]] = []
        for row in rows:
            entry = assignment[str(row["patient_id"])]
            if str(row.get("split")) != str(entry["split"]):
                raise ManifestError(f"patient {row['patient_id']} has split {row.get('split')} but {entry['split']} in the assignment")
            if _keeps(entry, fraction):
                kept.append(dict(row))
        _atomic_csv(destination, kept, list(rows[0].keys()))
        patients = sorted({(row["patient_id"], row["split"]) for row in kept})
        _atomic_csv(destination.parent / "train_patients.csv",
                    [{"patient_id": patient, "split": split} for patient, split in patients if split == "train"],
                    ("patient_id", "split"))
        counts = Counter(row["split"] for row in kept)
        return {"manifest": destination, "fraction": fraction, "rows_by_split": dict(counts),
                "train_patients": sum(1 for _, split in patients if split == "train")}


__all__ = ["ExperimentSplits", "build_assignment", "fraction_tag", "parse_fraction", "split_name"]
