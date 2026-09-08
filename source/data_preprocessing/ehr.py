"""Leakage-safe EHR readiness artifacts for a derived INSPECT dataset.

Stage 0 owns raw-event access.  It never invents a clinical score: it extracts the
MEDS/OMOP archive to a *local cache*, restricts events to the selected cohort, applies
the strict temporal boundary and prohibited-input policy, and writes reusable
case-level utilization features plus provenance/audit artifacts under ``clinical/``.

The fixed clinical variables used by a prognosis model remain an explicit clinical
contract.  They are deliberately not inferred from code descriptions here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tarfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .sources import StudyRecord


class EhrBuildError(RuntimeError):
    pass


EHR_FEATURE_COLUMNS = (
    "ehr_index_time_missing",
    "ehr_missing",
    "ehr_has_crosswalk",
    "ehr_events_prior",
    "ehr_events_365d",
    "ehr_events_30d",
    "ehr_numeric_events_prior",
)


@dataclass
class _CaseState:
    record: StudyRecord
    index_time: datetime | None
    events_prior: int = 0
    events_365d: int = 0
    events_30d: int = 0
    numeric_events_prior: int = 0

    def feature_row(self) -> dict[str, Any]:
        return {
            "ehr_index_time_missing": int(self.index_time is None),
            "ehr_missing": int(self.events_prior == 0),
            "ehr_has_crosswalk": int(self.record.has_ehr_crosswalk),
            "ehr_events_prior": self.events_prior,
            "ehr_events_365d": self.events_365d,
            "ehr_events_30d": self.events_30d,
            "ehr_numeric_events_prior": self.numeric_events_prior,
        }


@dataclass(frozen=True)
class EhrArtifacts:
    """Output summary and in-memory columns to merge into derived manifests."""

    status: str
    feature_columns: tuple[str, ...]
    by_study: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _load_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.dataset as ds  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment dependent
        raise EhrBuildError(
            "EHR preprocessing requires pyarrow; install requirements.txt before running stage 0"
        ) from exc
    return pa, pc, ds, pq


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text[:19] if "%H" in pattern else text[:10], pattern)
                    break
                except ValueError:
                    continue
            else:
                return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _as_patient_id(value: str) -> int:
    try:
        return int(str(value))
    except ValueError as exc:
        raise EhrBuildError(f"INSPECT EHR subject_id must be numeric, got {value!r}") from exc


def _archive_identity(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {"name": path.name, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def _safe_extract_archive(archive: Path, cache_root: Path, namespace: str) -> tuple[Path, dict[str, Any]]:
    """Extract the read-only parquet archive once into a local, reusable cache.

    The extracted raw event files are a cache rather than a derived dataset artifact:
    profile-specific outputs remain under ``/mnt/pe-storage`` while we avoid copying a
    multi-gigabyte source archive into every smoke/500/full profile.
    """
    if not archive.is_file():
        raise EhrBuildError(f"required INSPECT EHR archive is missing: {archive}")
    identity = _archive_identity(archive)
    destination = cache_root / namespace / archive.name.removesuffix(".tar.gz")
    marker = destination / ".pe-ehr-ready.json"
    expected = destination / "meds_omop_inspect"
    expected_data = expected / "data"
    expected_codes = expected / "metadata" / "codes.parquet"
    if marker.is_file() and expected_data.is_dir() and expected_codes.is_file():
        try:
            cached = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("archive_identity") == identity:
            return expected, {
                "status": "reused",
                "cache": str(expected),
                "archive": dict(cached.get("archive") or identity),
            }

    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    try:
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle:
                # MEDS has no need for symlinks/devices; rejecting them closes both traversal
                # and unexpected special-file hazards in a research data archive.
                if member.issym() or member.islnk() or member.isdev():
                    raise EhrBuildError(f"unsafe EHR archive member: {member.name}")
                target = (destination / member.name).resolve()
                try:
                    target.relative_to(destination_root)
                except ValueError as exc:
                    raise EhrBuildError(f"unsafe EHR archive member path: {member.name}") from exc
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise EhrBuildError(f"cannot read EHR archive member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except (tarfile.TarError, OSError) as exc:
        raise EhrBuildError(f"could not extract EHR archive {archive}: {exc}") from exc

    if not expected_data.is_dir() or not expected_codes.is_file():
        raise EhrBuildError(
            "EHR archive extracted but does not contain meds_omop_inspect/data and metadata/codes.parquet"
        )
    # Extraction already reads the complete multi-gigabyte archive.  The immutable raw
    # release stamp plus size/mtime is sufficient to invalidate this local cache; do not
    # immediately read it a second time merely to compute a checksum.
    fingerprint = dict(identity)
    _atomic_write_json(
        marker,
        {"archive": fingerprint, "archive_identity": identity, "format": "MEDS/OMOP parquet"},
    )
    return expected, {"status": "extracted", "cache": str(expected), "archive": fingerprint}


def _load_prohibited_policy(path: Path, code_descriptions: Mapping[str, str]) -> tuple[set[str], set[str], dict[str, Any]]:
    if not path.is_file():
        raise EhrBuildError(f"EHR prohibited-input policy is missing: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EhrBuildError(f"invalid EHR prohibited-input policy: {path}") from exc
    patterns = [re.compile(str(value), re.IGNORECASE) for value in payload.get("code_description_patterns", [])]
    prohibited_codes = {
        code
        for code, description in code_descriptions.items()
        if any(pattern.search(f"{code} {description}") for pattern in patterns)
    }
    prohibited_tables = {
        str(value).strip().lower() for value in payload.get("excluded_tables", []) if str(value).strip()
    }
    return prohibited_codes, prohibited_tables, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "excluded_tables": sorted(prohibited_tables),
        "code_description_patterns": [str(value) for value in payload.get("code_description_patterns", [])],
        "matched_code_count": len(prohibited_codes),
    }


def _load_code_descriptions(path: Path, pq: Any) -> dict[str, str]:
    if not path.is_file():
        raise EhrBuildError(f"MEDS code metadata is missing: {path}")
    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001 - pyarrow exceptions differ by version
        raise EhrBuildError(f"cannot read MEDS code metadata {path}: {type(exc).__name__}: {exc}") from exc
    fields = set(table.column_names)
    if "code" not in fields:
        raise EhrBuildError(f"MEDS code metadata has no 'code' column: {path}")
    codes = table.column("code").to_pylist()
    descriptions = table.column("description").to_pylist() if "description" in fields else [""] * len(codes)
    return {
        str(code): str(description or "")
        for code, description in zip(codes, descriptions)
        if code is not None
    }


def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(config or {})
    if float(data.get("end_before_ctpa_hours", 0.0)) < 0:
        raise EhrBuildError("ehr.end_before_ctpa_hours must be non-negative")
    windows = {int(value) for value in data.get("windows_days", [365, 30])}
    if not {365, 30}.issubset(windows):
        raise EhrBuildError("ehr.windows_days must include 365 and 30")
    if int(data.get("batch_size", 250000)) < 1:
        raise EhrBuildError("ehr.batch_size must be positive")
    if not str(data.get("archive") or "").strip():
        raise EhrBuildError("ehr.archive is required")
    if not str(data.get("prohibited_config") or "").strip():
        raise EhrBuildError("ehr.prohibited_config is required")
    return data


def _stream_events(
    data_directory: Path,
    cases_by_patient: Mapping[str, Sequence[_CaseState]],
    *,
    batch_size: int,
    end_before_ctpa_hours: float,
    prohibited_codes: set[str],
    prohibited_tables: set[str],
    train_vocabularies: Mapping[str, Counter[str]],
) -> tuple[int, Counter[str]]:
    pa, pc, ds, _ = _load_pyarrow()
    try:
        dataset = ds.dataset(data_directory, format="parquet")
    except Exception as exc:  # noqa: BLE001
        raise EhrBuildError(f"cannot open MEDS event dataset {data_directory}: {exc}") from exc
    available = set(dataset.schema.names)
    required = {"subject_id", "time", "code"}
    missing = sorted(required - available)
    if missing:
        raise EhrBuildError("MEDS event dataset lacks required columns: " + ", ".join(missing))
    requested = [name for name in ("subject_id", "time", "code", "numeric_value", "table", "clarity_table") if name in available]
    if not cases_by_patient:
        return 0, Counter()
    patient_ids = sorted({_as_patient_id(patient) for patient in cases_by_patient})
    try:
        scanner = dataset.scanner(
            columns=requested,
            filter=pc.is_in(ds.field("subject_id"), value_set=pa.array(patient_ids)),
            batch_size=batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        raise EhrBuildError(f"cannot scan MEDS events for selected patients: {exc}") from exc

    loaded_events = 0
    excluded: Counter[str] = Counter()
    boundary_delta = timedelta(hours=float(end_before_ctpa_hours))
    for batch in scanner.to_batches():
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            patient = str(columns["subject_id"][index])
            targets = cases_by_patient.get(patient)
            if not targets:
                continue
            loaded_events += 1
            event_time = _parse_time(columns.get("time", [None])[index])
            code = str(columns.get("code", [""])[index] or "")
            table_values = {
                str(columns.get(field, [""])[index] or "").strip().lower()
                for field in ("table", "clarity_table")
                if field in columns and str(columns[field][index] or "").strip()
            }
            numeric_value = columns.get("numeric_value", [None])[index]
            for case in targets:
                if case.index_time is None:
                    continue
                if event_time is None:
                    excluded["missing_or_invalid_event_time"] += 1
                    continue
                if event_time >= case.index_time - boundary_delta:
                    excluded["at_or_after_feature_end"] += 1
                    continue
                if table_values & prohibited_tables:
                    excluded["prohibited_table"] += 1
                    continue
                if code in prohibited_codes:
                    excluded["prohibited_code"] += 1
                    continue
                case.events_prior += 1
                if numeric_value is not None and not (
                    isinstance(numeric_value, float) and math.isnan(numeric_value)
                ):
                    case.numeric_events_prior += 1
                if event_time >= case.index_time - timedelta(days=365):
                    case.events_365d += 1
                if event_time >= case.index_time - timedelta(days=30):
                    case.events_30d += 1
                if case.record.split == "train" and code:
                    train_vocabularies["prior"][code] += 1
                    if event_time >= case.index_time - timedelta(days=365):
                        train_vocabularies["window365"][code] += 1
                    if event_time >= case.index_time - timedelta(days=30):
                        train_vocabularies["window30"][code] += 1
    return loaded_events, excluded


def _vocabulary_rows(
    counters: Mapping[str, Counter[str]], limits: Mapping[str, Any], descriptions: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in ("prior", "window365", "window30"):
        limit = int(limits.get(window, 0))
        if limit < 1:
            raise EhrBuildError(f"ehr.train_vocabulary.{window} must be positive")
        for rank, (code, count) in enumerate(counters[window].most_common(limit), start=1):
            rows.append(
                {
                    "window": window,
                    "rank": rank,
                    "code": code,
                    "description": descriptions.get(code, ""),
                    "train_case_event_count": count,
                }
            )
    return rows


def build_ehr_readiness(
    records: Sequence[StudyRecord],
    *,
    release_root: Path,
    cache_root: Path,
    output_dir: Path,
    config: Mapping[str, Any] | None,
    code_root: Path,
) -> EhrArtifacts:
    """Build safe, patient-linked pre-CTPA EHR readiness artifacts for one profile."""
    settings = _settings(config)
    if not bool(settings.get("enabled", True)):
        metadata = {"status": "disabled", "reason": "ehr.enabled=false"}
        _atomic_write_json(output_dir / "ehr_status.json", metadata)
        return EhrArtifacts("disabled", (), {}, metadata)
    if not records:
        raise EhrBuildError("cannot build EHR readiness for an empty cohort")

    archive = release_root / "EHR" / str(settings["archive"])
    namespace = str(settings.get("cache_namespace") or "inspect_meds_omop")
    source_root, extraction = _safe_extract_archive(archive, cache_root, namespace)
    _, _, _, pq = _load_pyarrow()
    descriptions = _load_code_descriptions(source_root / "metadata" / "codes.parquet", pq)
    policy_path = Path(str(settings["prohibited_config"]))
    if not policy_path.is_absolute():
        policy_path = code_root / policy_path
    prohibited_codes, prohibited_tables, policy = _load_prohibited_policy(policy_path, descriptions)

    states = [_CaseState(record=record, index_time=_parse_time(record.procedure_datetime)) for record in records]
    cases_by_patient: dict[str, list[_CaseState]] = defaultdict(list)
    for case in states:
        # The EHR parquet identifier is only trusted for study rows that the release's
        # image/EHR crosswalk explicitly links. A non-linked row stays in the cohort but
        # receives EHR-missing features rather than an inferred patient join.
        if case.record.has_ehr_crosswalk:
            cases_by_patient[case.record.patient_id].append(case)
    vocabularies = {"prior": Counter(), "window365": Counter(), "window30": Counter()}
    loaded_events, excluded = _stream_events(
        source_root / "data",
        cases_by_patient,
        batch_size=int(settings.get("batch_size", 250000)),
        end_before_ctpa_hours=float(settings.get("end_before_ctpa_hours", 0.0)),
        prohibited_codes=prohibited_codes,
        prohibited_tables=prohibited_tables,
        train_vocabularies=vocabularies,
    )

    rows = []
    by_study: dict[str, dict[str, Any]] = {}
    for case in states:
        features = case.feature_row()
        if case.record.study_id in by_study:
            raise EhrBuildError(f"duplicate study id in EHR cohort: {case.record.study_id}")
        by_study[case.record.study_id] = features
        rows.append(
            {
                "patient_id": case.record.patient_id,
                "study_id": case.record.study_id,
                "split": case.record.split,
                "index_time": case.record.procedure_datetime,
                **features,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table = output_dir / "ehr_features.csv"
    vocabulary_table = output_dir / "ehr_train_vocabulary.csv"
    metadata_path = output_dir / "ehr_metadata.json"
    _atomic_write_csv(feature_table, rows, ["patient_id", "study_id", "split", "index_time", *EHR_FEATURE_COLUMNS])
    vocabulary_rows = _vocabulary_rows(vocabularies, dict(settings.get("train_vocabulary") or {}), descriptions)
    _atomic_write_csv(
        vocabulary_table,
        vocabulary_rows,
        ["window", "rank", "code", "description", "train_case_event_count"],
    )
    metadata = {
        "status": "built",
        "format": "MEDS/OMOP parquet",
        "extraction": extraction,
        "cases": len(states),
        "patients": len({case.record.patient_id for case in states}),
        "crosswalk_linked_patients": len(cases_by_patient),
        "crosswalk_linked_cases": sum(case.record.has_ehr_crosswalk for case in states),
        "crosswalk_missing_cases": sum(not case.record.has_ehr_crosswalk for case in states),
        "patients_with_permitted_pre_ctpa_events": len({
            case.record.patient_id for case in states if case.events_prior > 0
        }),
        "missing_index_time": sum(case.index_time is None for case in states),
        "ehr_missing_cases": sum(case.events_prior == 0 for case in states),
        "loaded_cohort_events": loaded_events,
        "excluded_case_event_counts": dict(sorted(excluded.items())),
        "end_before_ctpa_hours": float(settings.get("end_before_ctpa_hours", 0.0)),
        "feature_end_operator": "event_time < procedure_datetime - end_before_ctpa_hours",
        "prohibited_policy": policy,
        "train_only_vocabulary": True,
        "vocabulary_counts": {name: len(counter) for name, counter in vocabularies.items()},
        "feature_columns": list(EHR_FEATURE_COLUMNS),
        "feature_table": str(feature_table),
        "vocabulary_table": str(vocabulary_table),
    }
    _atomic_write_json(metadata_path, metadata)
    return EhrArtifacts("built", EHR_FEATURE_COLUMNS, by_study, metadata)
