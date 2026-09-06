from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    pass


REQUIRED_COLUMNS = ("patient_id", "study_id", "split")
ALLOWED_SPLITS = ("train", "validation", "test", "external")
SILVER_STATUSES = ("accepted", "abstained", "no_result")


def _normal(value: Any) -> str:
    return "" if value is None else str(value).strip()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise ManifestError(f"manifest not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ManifestError("pandas and pyarrow are required for Parquet manifests") from exc
        return pd.read_parquet(source).to_dict(orient="records")
    raise ManifestError(f"unsupported manifest format: {source.suffix}")


@dataclass(frozen=True)
class ManifestAudit:
    path: str
    rows: int
    patients: int
    studies: int
    split_patients: dict[str, int]
    split_studies: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    duplicate_study_ids: tuple[str, ...]
    duplicate_patient_studies: tuple[str, ...]
    patient_overlap: dict[str, tuple[str, ...]]
    patient_fold_overlap: dict[str, tuple[str, ...]]
    missing_files: tuple[str, ...]
    missing_primary_labels: tuple[str, ...]
    invalid_numeric_values: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def audit_manifest(
    path: str | Path,
    label_columns: Sequence[str] = (),
    file_column: str | None = "image_path",
    data_root: str | Path | None = None,
    split_aliases: Mapping[str, str] | None = None,
    required_columns: Sequence[str] = (),
    file_columns: Sequence[str] = (),
    numeric_columns: Sequence[str] = (),
    primary_target: str | None = None,
    patient_column: str = "patient_id",
    study_column: str = "study_id",
    split_column: str = "split",
    fold_column: str | None = None,
    required_splits: Sequence[str] = (),
) -> ManifestAudit:
    source = Path(path)
    rows = read_rows(source)
    errors: list[str] = []
    if not rows:
        errors.append("manifest is empty")
    columns = set(rows[0]) if rows else set()
    identity_columns = (patient_column, study_column, split_column)
    missing_columns = [column for column in identity_columns if column not in columns]
    missing_columns.extend(column for column in required_columns if column and column not in columns and column not in missing_columns)
    if missing_columns:
        errors.append("missing required columns: " + ", ".join(missing_columns))
    aliases = dict(split_aliases or {})
    patient_splits: dict[str, set[str]] = defaultdict(set)
    patient_folds: dict[str, set[str]] = defaultdict(set)
    split_patients: dict[str, set[str]] = defaultdict(set)
    split_studies: Counter[str] = Counter()
    study_ids: list[str] = []
    pairs: list[str] = []
    class_counts: dict[str, Counter[str]] = {column: Counter() for column in label_columns}
    missing_files: list[str] = []
    missing_primary: list[str] = []
    invalid_numeric: list[str] = []
    checked_file_columns = tuple(
        dict.fromkeys((*(file_columns or ()), *((file_column,) if file_column else ())))
    )
    root = Path(data_root).resolve() if data_root else source.parent.resolve()
    for index, row in enumerate(rows, start=2):
        patient = _normal(row.get(patient_column))
        study = _normal(row.get(study_column))
        raw_split = _normal(row.get(split_column)).lower()
        split = aliases.get(raw_split, raw_split)
        if not patient or not study or not split:
            errors.append(f"row {index} has empty patient_id, study_id, or split")
            continue
        if split not in ALLOWED_SPLITS:
            errors.append(f"row {index} has unsupported split {raw_split!r}")
        patient_splits[patient].add(split)
        if fold_column:
            fold = _normal(row.get(fold_column))
            if not fold:
                errors.append(f"row {index} has empty {fold_column}")
            else:
                patient_folds[patient].add(fold)
        split_patients[split].add(patient)
        split_studies[split] += 1
        study_ids.append(study)
        pairs.append(patient + "\x00" + study)
        for column in label_columns:
            class_counts[column][_normal(row.get(column)) or "missing"] += 1
        if primary_target and not _normal(row.get(primary_target)):
            missing_primary.append(f"{patient}/{study}")
        for column in numeric_columns:
            raw_numeric = _normal(row.get(column))
            if not raw_numeric:
                continue
            try:
                value = float(raw_numeric)
            except (TypeError, ValueError):
                invalid_numeric.append(f"row:{index}:{column}:{raw_numeric}")
            else:
                if not math.isfinite(value):
                    invalid_numeric.append(f"row:{index}:{column}:{raw_numeric}")
        for column in checked_file_columns:
            raw_file = _normal(row.get(column))
            if not raw_file:
                missing_files.append(f"row:{index}:{column}:empty")
            else:
                file_path = Path(raw_file)
                if not file_path.is_absolute():
                    file_path = root / file_path
                if not file_path.is_file():
                    missing_files.append(f"{column}:{file_path}")
    duplicate_studies = tuple(sorted(key for key, count in Counter(study_ids).items() if count > 1))
    duplicate_pairs = tuple(sorted(key.replace("\x00", "/") for key, count in Counter(pairs).items() if count > 1))
    overlap = {
        patient: tuple(sorted(splits))
        for patient, splits in sorted(patient_splits.items())
        if len(splits) > 1
    }
    fold_overlap = {
        patient: tuple(sorted(folds))
        for patient, folds in sorted(patient_folds.items())
        if len(folds) > 1
    }
    if duplicate_studies:
        errors.append(f"duplicate study_id values: {len(duplicate_studies)}")
    if duplicate_pairs:
        errors.append(f"duplicate patient/study rows: {len(duplicate_pairs)}")
    if overlap:
        errors.append(f"patient overlap across splits: {len(overlap)}")
    if fold_overlap:
        errors.append(f"patient assigned to multiple folds: {len(fold_overlap)}")
    absent_splits = sorted(set(required_splits) - set(split_studies))
    if absent_splits:
        errors.append("missing required splits: " + ", ".join(absent_splits))
    if missing_files:
        errors.append(f"missing required files: {len(missing_files)}")
    if missing_primary:
        errors.append(f"missing primary target {primary_target}: {len(missing_primary)}")
    if invalid_numeric:
        errors.append(f"invalid numeric values: {len(invalid_numeric)}")
    return ManifestAudit(
        path=str(source.resolve()),
        rows=len(rows),
        patients=len(patient_splits),
        studies=len(set(study_ids)),
        split_patients={key: len(value) for key, value in sorted(split_patients.items())},
        split_studies=dict(sorted(split_studies.items())),
        class_distribution={key: dict(sorted(value.items())) for key, value in class_counts.items()},
        duplicate_study_ids=duplicate_studies,
        duplicate_patient_studies=duplicate_pairs,
        patient_overlap=overlap,
        patient_fold_overlap=fold_overlap,
        missing_files=tuple(missing_files),
        missing_primary_labels=tuple(missing_primary),
        invalid_numeric_values=tuple(invalid_numeric),
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class TableAudit:
    path: str
    rows: int
    columns: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def audit_report_table(
    path: str | Path,
    *,
    patient_column: str = "patient_id",
    study_column: str = "study_id",
    report_column: str = "report_id",
    text_column: str = "report_text",
    split_column: str | None = None,
) -> TableAudit:
    """Validate the report-mining input without requiring report text to be non-empty.

    Empty text is a valid source row because the silver pipeline must emit ``no_result``
    rather than invent a label.  Identifiers, schema, split values, and report keys are
    nevertheless strict.
    """
    source = Path(path)
    rows = read_rows(source)
    columns = tuple(sorted({key for row in rows for key in row}))
    required = [patient_column, study_column, report_column, text_column]
    if split_column:
        required.append(split_column)
    errors: list[str] = []
    missing = [name for name in required if name not in columns]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))
    if not rows:
        errors.append("report table is empty")
    keys: list[str] = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows, start=2):
        identifiers = [_normal(row.get(name)) for name in (patient_column, study_column, report_column)]
        if not all(identifiers):
            errors.append(f"row {index} has an empty patient/study/report identifier")
            continue
        keys.append(identifiers[2])
        if split_column:
            split = _normal(row.get(split_column)).lower()
            if split not in ALLOWED_SPLITS:
                errors.append(f"row {index} has unsupported split {split!r}")
            patient_splits[identifiers[0]].add(split)
    duplicates = tuple(sorted(key for key, count in Counter(keys).items() if count > 1))
    if duplicates:
        errors.append(f"duplicate report_id values: {len(duplicates)}")
    overlap = sum(len(values) > 1 for values in patient_splits.values())
    if overlap:
        errors.append(f"patient overlap across report splits: {overlap}")
    return TableAudit(str(source.resolve()), len(rows), columns, duplicates, tuple(errors))


def audit_silver_table(
    path: str | Path,
    *,
    allowed_targets: Sequence[str] = (),
) -> TableAudit:
    """Validate the compact long-form auxiliary-label training artifact."""
    source = Path(path)
    rows = read_rows(source)
    columns = tuple(sorted({key for row in rows for key in row}))
    required = ("patient_id", "study_id", "report_id", "target", "value", "status", "source")
    errors: list[str] = []
    missing = [name for name in required if name not in columns]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))
    if not rows:
        errors.append("silver-label table is empty")
    target_set = set(allowed_targets)
    keys: list[str] = []
    for index, row in enumerate(rows, start=2):
        identifiers = [_normal(row.get(name)) for name in ("patient_id", "study_id", "report_id")]
        target = _normal(row.get("target"))
        status = _normal(row.get("status"))
        value = row.get("value")
        value_missing = value is None or _normal(value) in {"", "null", "None"}
        if not all(identifiers) or not target:
            errors.append(f"row {index} has an empty identifier or target")
            continue
        keys.append(identifiers[2] + "\x00" + target)
        if target_set and target not in target_set:
            errors.append(f"row {index} has unsupported target {target!r}")
        if status not in SILVER_STATUSES:
            errors.append(f"row {index} has unsupported status {status!r}")
        elif status == "accepted" and value_missing:
            errors.append(f"row {index} accepted target has no value")
        elif status != "accepted" and not value_missing:
            errors.append(f"row {index} non-accepted target preserves a final value")
        confidence = row.get("confidence")
        if confidence is not None and _normal(confidence):
            try:
                numeric = float(confidence)
            except (TypeError, ValueError):
                errors.append(f"row {index} confidence is not numeric")
            else:
                if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                    errors.append(f"row {index} confidence is outside [0,1]")
    duplicates = tuple(
        sorted(key.replace("\x00", "/") for key, count in Counter(keys).items() if count > 1)
    )
    if duplicates:
        errors.append(f"duplicate report/target rows: {len(duplicates)}")
    return TableAudit(str(source.resolve()), len(rows), columns, duplicates, tuple(errors))


def patient_ids_for_splits(
    rows: Iterable[Mapping[str, Any]],
    splits: Iterable[str],
    *,
    patient_column: str = "patient_id",
    split_column: str = "split",
) -> set[str]:
    selected = {str(value).strip().lower() for value in splits}
    return {
        _normal(row.get(patient_column))
        for row in rows
        if _normal(row.get(split_column)).lower() in selected and _normal(row.get(patient_column))
    }


def require_valid_manifest(*args: Any, **kwargs: Any) -> ManifestAudit:
    audit = audit_manifest(*args, **kwargs)
    if not audit.ok:
        raise ManifestError("; ".join(audit.errors))
    return audit


def create_patient_split(
    rows: Iterable[Mapping[str, Any]],
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> list[dict[str, Any]]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ManifestError("train/validation/test ratios must be positive and sum to one")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        patient = _normal(row.get("patient_id"))
        if not patient:
            raise ManifestError("cannot split a row without patient_id")
        if _normal(row.get("split")):
            raise ManifestError("refusing to replace an existing split assignment")
        grouped[patient].append(dict(row))
    patients = sorted(grouped)
    random.Random(int(seed)).shuffle(patients)
    train_end = round(len(patients) * ratios[0])
    validation_end = train_end + round(len(patients) * ratios[1])
    assignments = {
        patient: ("train" if index < train_end else "validation" if index < validation_end else "test")
        for index, patient in enumerate(patients)
    }
    output: list[dict[str, Any]] = []
    for patient in sorted(grouped):
        for row in grouped[patient]:
            row["split"] = assignments[patient]
            output.append(row)
    return output


def audit_temporal_holdout(
    rows: Sequence[Mapping[str, Any]],
    *,
    date_column: str,
    target_column: str,
    minimum_events_per_split: int = 1,
    patient_column: str = "patient_id",
    split_column: str = "split",
) -> dict[str, Any]:
    """Validate chronological patient splits and event support without creating a split."""

    if minimum_events_per_split < 1:
        raise ManifestError("minimum_events_per_split must be positive")
    dates: dict[str, list[datetime]] = defaultdict(list)
    events: dict[str, set[str]] = defaultdict(set)
    patients: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    for index, row in enumerate(rows, start=2):
        split = _normal(row.get(split_column)).lower()
        patient = _normal(row.get(patient_column))
        raw_date = _normal(row.get(date_column))
        if split not in {"train", "validation", "test"}:
            continue
        try:
            parsed = datetime.fromisoformat(raw_date)
        except (TypeError, ValueError):
            errors.append(f"row {index} has invalid {date_column}: {raw_date!r}")
            continue
        dates[split].append(parsed)
        patients[split].add(patient)
        try:
            event = float(row.get(target_column))
        except (TypeError, ValueError):
            errors.append(f"row {index} has invalid {target_column}")
        else:
            if event == 1:
                events[split].add(patient)
    for split in ("train", "validation", "test"):
        if not dates[split]:
            errors.append(f"temporal holdout has no dated rows for {split}")
        if len(events[split]) < minimum_events_per_split:
            errors.append(
                f"temporal holdout {split} has {len(events[split])} events; "
                f"minimum={minimum_events_per_split}"
            )
    if all(dates[split] for split in ("train", "validation", "test")):
        if max(dates["train"]) > min(dates["validation"]):
            errors.append("training dates overlap or follow validation dates")
        if max(dates["validation"]) > min(dates["test"]):
            errors.append("validation dates overlap or follow test dates")
    return {
        "ok": not errors,
        "date_column": date_column,
        "target_column": target_column,
        "minimum_events_per_split": minimum_events_per_split,
        "patients": {name: len(patients[name]) for name in ("train", "validation", "test")},
        "events": {name: len(events[name]) for name in ("train", "validation", "test")},
        "date_ranges": {
            name: {
                "minimum": min(dates[name]).isoformat() if dates[name] else None,
                "maximum": max(dates[name]).isoformat() if dates[name] else None,
            }
            for name in ("train", "validation", "test")
        },
        "errors": errors,
    }
