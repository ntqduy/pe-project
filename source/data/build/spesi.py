"""sPESI materialisation from the raw MEDS/OMOP events.

The score is computed here rather than read from a curated table, because the INSPECT
release ships the events sPESI needs and scores them itself. What this module keeps from
the previous PESI gate is the part that mattered: every component is resolved from an
explicit, logged code list, a case missing any component is left unscored, and the
effective code set lands in an audit next to the features.

Component definitions, thresholds and vital codes come from
``third_party/repos/INSPECT_public/ehr/4_compute_pesi_score.py``; see
``configs/clinical/spesi_mapping.yaml`` for the one deliberate deviation (ICD-10 prefix
expansion in place of FEMR ontology expansion).
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from source.clinical.spesi import (
    DEFAULT_FIELDS,
    SPESI_COMPONENTS,
    SPESI_FEATURE_COLUMNS,
    compute_spesi,
)

from .ehr import (
    _as_patient_id,
    _load_code_descriptions,
    _load_pyarrow,
    _parse_time,
    canonical_patient_id,
    group_by_canonical_patient,
)
from .sources import StudyRecord


class SpesiBuildError(RuntimeError):
    pass


# Accepted spellings of each mapping ``expected_unit``, compared after _normalize_unit
# (lower-case, inner whitespace collapsed). An event whose unit column is empty is accepted
# unvalidated; a non-empty unit outside this set is skipped and counted as
# ``unit_mismatch_<component>:<unit>`` so a new spelling is visible, not lost.
UNIT_ALIASES = {
    "/min": {
        "/min",
        "1/min",
        "/minute",
        "per min",
        "per minute",
        "bpm",
        "beats/min",
        "beats/minute",
        "beats per minute",
        "{beats}/min",
        "{beat}/min",
        "counts per minute",
    },
    "mmhg": {"mmhg", "mm[hg]", "mm hg", "millimeter mercury column"},
    "%": {"%", "percent", "pct"},
}
# Conservative physiologic sanity ranges, (low, high] in the expected unit. A value outside
# is an entry error, not a measurement, so that event is skipped (counted as
# ``implausible_<component>``) and an earlier plausible event can still supply the vital.
# SpO2 recorded as a fraction (0, 1] is converted to percent first. An SpO2 event whose unit
# says percent must also reach SPO2_PERCENT_FLOOR: below it the value is not a plausible
# reading for a living ED patient (a unitless value keeps the wider range).
PLAUSIBLE_RANGES = {
    "pulse": (0.0, 300.0),
    "systolic_bp": (0.0, 300.0),
    "oxygen_saturation": (0.0, 100.0),
}
SPO2_PERCENT_FLOOR = 50.0


def _normalize_unit(unit: Any) -> str:
    return " ".join(str(unit or "").split()).lower()


def _plausible_vital(component: str, number: float, unit: str = "") -> float | None:
    if component == "oxygen_saturation" and 0.0 < number <= 1.0:
        number *= 100.0
    low, high = PLAUSIBLE_RANGES.get(component, (-math.inf, math.inf))
    if not low < number <= high:
        return None
    if (
        component == "oxygen_saturation"
        and _normalize_unit(unit) in UNIT_ALIASES["%"]
        and number < SPO2_PERCENT_FLOOR
    ):
        return None
    return number


@dataclass(frozen=True)
class SpesiArtifacts:
    """sPESI provenance plus the feature columns safe to merge into manifests."""

    status: str
    feature_columns: tuple[str, ...]
    by_study: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]
    components_by_study: Mapping[str, Mapping[str, Any]] | None = None


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency environment specific
        raise SpesiBuildError("sPESI mapping requires PyYAML; install requirements.txt") from exc
    if not path.is_file():
        raise SpesiBuildError(f"sPESI mapping configuration is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # YAML parser error types are version dependent
        raise SpesiBuildError(f"invalid sPESI mapping configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SpesiBuildError(f"sPESI mapping must contain a mapping: {path}")
    return payload


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        raise SpesiBuildError(f"sPESI code list must be a string/list, got {type(value).__name__}")
    return [str(item).strip() for item in value if str(item).strip()]


def _resolve_mapping_path(value: Any, code_root: Path) -> Path:
    text = os.path.expandvars(str(value or "").strip())
    if not text or "${" in text:
        raise SpesiBuildError("spesi.mapping_config must be a resolved path")
    path = Path(text)
    return path if path.is_absolute() else code_root / path


def _ehr_source_root(ehr_metadata: Mapping[str, Any]) -> Path:
    extraction = dict(ehr_metadata.get("extraction") or {})
    source_root = Path(str(extraction.get("cache") or ""))
    if not (source_root / "metadata" / "codes.parquet").is_file():
        raise SpesiBuildError(
            "cannot resolve sPESI source codes because EHR extraction metadata does not point to codes.parquet"
        )
    return source_root


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def resolve_code_sets(
    mapping: Mapping[str, Any], catalog: Mapping[str, str]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Expand each component's literal codes and prefixes against the release catalog.

    Returns the effective code list per component plus an audit recording which literal
    codes were absent from the catalog and how many codes each prefix matched. A prefix
    that matches nothing is reported, never silently dropped.
    """
    components = dict(mapping.get("components") or {})
    missing = sorted(set(SPESI_COMPONENTS) - set(components))
    if missing:
        raise SpesiBuildError("sPESI mapping is missing component(s): " + ", ".join(missing))
    code_sets: dict[str, list[str]] = {}
    audit: dict[str, Any] = {}
    known = set(catalog)
    for component in SPESI_COMPONENTS:
        item = dict(components[component] or {})
        literal = _as_string_list(item.get("source_codes"))
        prefixes = _as_string_list(item.get("source_code_prefixes"))
        expanded: dict[str, list[str]] = {}
        for prefix in prefixes:
            expanded[prefix] = sorted(code for code in known if code.startswith(prefix))
        effective = list(dict.fromkeys([*literal, *(c for codes in expanded.values() for c in codes)]))
        code_sets[component] = effective
        audit[component] = {
            "type": str(item.get("type") or ""),
            "literal_codes": literal,
            "literal_codes_absent_from_catalog": sorted(set(literal) - known - {"MEDS_BIRTH"}),
            "prefixes": {prefix: len(codes) for prefix, codes in expanded.items()},
            "prefixes_matching_nothing": sorted(p for p, codes in expanded.items() if not codes),
            "effective_code_count": len(effective),
            "effective_codes": effective[:200],
            "absent_means_false": bool(item.get("absent_means_false", False)),
        }
    return code_sets, audit


def _select_components(
    records: Sequence[StudyRecord],
    *,
    mapping: Mapping[str, Any],
    code_sets: Mapping[str, Sequence[str]],
    source_root: Path,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Resolve each study's six components from the MEDS event stream."""
    components = {name: dict(dict(mapping.get("components") or {}).get(name) or {}) for name in SPESI_COMPONENTS}
    index = dict(mapping.get("index") or {})
    after = timedelta(hours=float(index.get("event_window_after_index_hours") or 0))
    # null/absent = unlimited lookback; 0 is a real (empty) window, never "unlimited".
    vital_hours = index.get("vital_lookback_hours")
    vital_window = None if vital_hours in (None, "") else timedelta(hours=float(vital_hours))
    # INSPECT compares a fractional age against 80; the clinical convention is completed
    # years. The two disagree for anyone between their 80th and 81st birthday, so this is
    # an explicit contract rather than an implementation detail.
    age_precision = str(index.get("age_precision") or "completed_years").strip().lower()
    if age_precision not in {"completed_years", "fractional"}:
        raise SpesiBuildError("index.age_precision must be completed_years or fractional")
    days_per_year = float(index.get("age_days_per_year") or 365.2425)
    if days_per_year <= 0:
        raise SpesiBuildError("index.age_days_per_year must be positive")
    comorbidity_hours = index.get("comorbidity_lookback_hours")
    comorbidity_window = None if comorbidity_hours in (None, "") else timedelta(hours=float(comorbidity_hours))

    by_patient = group_by_canonical_patient(
        (record for record in records if record.has_ehr_crosswalk),
        lambda record: record.patient_id,
    )

    code_to_components: dict[str, list[str]] = defaultdict(list)
    for component, codes in code_sets.items():
        for code in codes:
            code_to_components[code].append(component)

    selected: dict[str, dict[str, Any]] = {record.study_id: {} for record in records}
    skipped: Counter[str] = Counter()
    if not by_patient or not code_to_components:
        return selected, dict(skipped)

    pa, pc, dataset_module, _ = _load_pyarrow()
    try:
        dataset = dataset_module.dataset(source_root / "data", format="parquet")
    except Exception as exc:  # pyarrow exceptions differ by version
        raise SpesiBuildError(f"cannot open MEDS event dataset for sPESI: {exc}") from exc
    available = set(dataset.schema.names)
    absent = sorted({"subject_id", "time", "code"} - available)
    if absent:
        raise SpesiBuildError("MEDS event dataset lacks sPESI columns: " + ", ".join(absent))
    columns = [n for n in ("subject_id", "time", "code", "numeric_value", "unit") if n in available]
    patient_ids = sorted({_as_patient_id(patient) for patient in by_patient})
    try:
        expression = (
            pc.is_in(dataset_module.field("subject_id"), value_set=pa.array(patient_ids))
            & pc.is_in(dataset_module.field("code"), value_set=pa.array(sorted(code_to_components)))
        )
        scanner = dataset.scanner(columns=columns, filter=expression, batch_size=batch_size)
    except Exception as exc:  # pyarrow exceptions differ by version
        raise SpesiBuildError(f"cannot scan MEDS sPESI events: {exc}") from exc

    for batch in scanner.to_batches():
        values = batch.to_pydict()
        for row in range(batch.num_rows):
            subject = values["subject_id"][row]
            cases = () if subject is None else by_patient.get(canonical_patient_id(subject)) or ()
            if not cases:
                continue
            code = str(values["code"][row] or "")
            event_time = _parse_time(values["time"][row])
            if event_time is None:
                skipped["missing_or_invalid_event_time"] += len(cases)
                continue
            numeric_value = values.get("numeric_value", [None])[row]
            raw_unit = str(values.get("unit", [None])[row] or "").strip()
            for record in cases:
                index_time = _parse_time(record.procedure_datetime)
                if index_time is None:
                    skipped["missing_index_time"] += 1
                    continue
                if event_time >= index_time + after:
                    skipped["after_index_window"] += 1
                    continue
                for component in code_to_components[code]:
                    kind = str(components[component].get("type") or "").strip().lower()
                    if kind == "numeric":
                        if vital_window is not None and event_time < index_time - vital_window:
                            skipped["outside_vital_lookback"] += 1
                            continue
                        expected_unit = _normalize_unit(components[component].get("expected_unit"))
                        accepted = UNIT_ALIASES.get(expected_unit, {expected_unit})
                        if expected_unit and raw_unit and _normalize_unit(raw_unit) not in accepted:
                            skipped[f"unit_mismatch_{component}:{raw_unit}"] += 1
                            continue
                        number = _finite_number(numeric_value)
                        if number is None:
                            skipped[f"non_numeric_{component}"] += 1
                            continue
                        number = _plausible_vital(component, number, raw_unit)
                        if number is None:
                            skipped[f"implausible_{component}"] += 1
                            continue
                        entry = {"value": number, "source_code": code, "event_time": event_time, "unit": raw_unit}
                    elif kind == "boolean":
                        if comorbidity_window is not None and event_time < index_time - comorbidity_window:
                            skipped["outside_comorbidity_lookback"] += 1
                            continue
                        entry = {"value": True, "source_code": code, "event_time": event_time, "unit": ""}
                    elif kind == "age_years":
                        years = (index_time - event_time).days / days_per_year
                        if not math.isfinite(years) or years < 0:
                            skipped["implausible_age"] += 1
                            continue
                        age = years if age_precision == "fractional" else int(years)
                        entry = {"value": age, "source_code": code, "event_time": event_time, "unit": "years"}
                    else:
                        skipped[f"unsupported_type_{component}"] += 1
                        continue
                    previous = selected[record.study_id].get(component)
                    # Latest eligible event wins for vitals and age; for a comorbidity any
                    # single hit is already decisive, so the first one is kept.
                    if previous is None or (kind != "boolean" and event_time > previous["event_time"]):
                        selected[record.study_id][component] = entry
    return selected, dict(skipped)


def build_spesi_artifacts(
    records: Sequence[StudyRecord],
    *,
    output_dir: Path,
    config: Mapping[str, Any] | None,
    ehr_metadata: Mapping[str, Any],
    code_root: Path,
) -> SpesiArtifacts:
    """Resolve, score and write sPESI for one built cohort."""
    settings = dict(config or {})
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(settings.get("enabled", True)):
        metadata = {"status": "disabled", "reason": "spesi.enabled=false"}
        _write_json(output_dir / "spesi_status.json", metadata)
        return SpesiArtifacts("disabled", (), {}, metadata)
    ehr_status = str(ehr_metadata.get("status") or "missing")
    if ehr_status != "built":
        # sPESI is scored from the MEDS events, so without a built EHR it is unavailable
        # (has_spesi=0 for every case; preflight reports the reason), not a build failure.
        metadata = {
            "status": "unavailable",
            "reason": f"EHR status is {ehr_status!r}; sPESI needs the extracted MEDS events",
            "studies": len(records),
            "scored_cases": 0,
            "scored_cases_by_split": {},
        }
        _write_json(output_dir / "spesi_status.json", metadata)
        return SpesiArtifacts("unavailable", (), {}, metadata)

    mapping_path = _resolve_mapping_path(settings.get("mapping_config"), code_root)
    mapping = _read_yaml(mapping_path)
    # `spesi.index_overrides` layers onto the mapping's index block so a run can switch
    # to INSPECT's conventions without a second copy of the contract. The resolved index
    # is written into spesi_status.json, so which rules produced a score is never implicit.
    overrides = dict(settings.get("index_overrides") or {})
    if overrides:
        unknown = sorted(set(overrides) - set(mapping.get("index") or {}))
        if unknown:
            raise SpesiBuildError(
                "spesi.index_overrides names key(s) absent from the mapping index: "
                + ", ".join(unknown)
            )
        mapping = {**mapping, "index": {**dict(mapping.get("index") or {}), **overrides}}
    source_root = _ehr_source_root(ehr_metadata)
    _, _, _, parquet = _load_pyarrow()
    catalog = _load_code_descriptions(source_root / "metadata" / "codes.parquet", parquet)
    code_sets, code_audit = resolve_code_sets(mapping, catalog)

    batch_size = int(settings.get("batch_size", 250000))
    if batch_size < 1:
        raise SpesiBuildError("spesi.batch_size must be positive")
    selected, skipped = _select_components(
        records, mapping=mapping, code_sets=code_sets, source_root=source_root, batch_size=batch_size
    )

    components_cfg = dict(mapping.get("components") or {})
    absent_false = {
        name for name in SPESI_COMPONENTS if bool(dict(components_cfg.get(name) or {}).get("absent_means_false"))
    }
    component_fields = ["patient_id", "study_id", "split"]
    for name in SPESI_COMPONENTS:
        component_fields += [name, f"{name}_missing", f"{name}_source_code", f"{name}_event_time"]

    component_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    by_study: dict[str, dict[str, Any]] = {}
    components_by_study: dict[str, dict[str, Any]] = {}
    missing_counts: Counter[str] = Counter()

    for record in records:
        found = selected.get(record.study_id) or {}
        values: dict[str, Any] = {}
        row: dict[str, Any] = {
            "patient_id": record.patient_id,
            "study_id": record.study_id,
            "split": record.split,
        }
        for name in SPESI_COMPONENTS:
            entry = found.get(name)
            if entry is None and name in absent_false:
                # Absence of a comorbidity code is a negative, as in INSPECT's scorer.
                entry = {"value": False, "source_code": "", "event_time": "", "unit": ""}
            value = None if entry is None else entry["value"]
            values[name] = value
            if value is None:
                missing_counts[name] += 1
            row[name] = "" if value is None else (int(value) if isinstance(value, bool) else value)
            row[f"{name}_missing"] = int(value is None)
            row[f"{name}_source_code"] = "" if entry is None else entry["source_code"]
            row[f"{name}_event_time"] = "" if entry is None else entry["event_time"]
        component_rows.append(row)
        components_by_study[record.study_id] = dict(values)

        result = compute_spesi(values, fields=DEFAULT_FIELDS)
        features = {
            "spesi": result.spesi_score,
            "spesi_high_risk": None if result.high_risk is None else int(result.high_risk),
            "spesi_computable": int(result.computable),
        }
        by_study[record.study_id] = features
        feature_rows.append(
            {"patient_id": record.patient_id, "study_id": record.study_id, "split": record.split, **features}
        )

    components_path = _write_csv(output_dir / "spesi_components.csv", component_rows, component_fields)
    features_path = _write_csv(
        output_dir / "spesi_features.csv",
        feature_rows,
        ["patient_id", "study_id", "split", *SPESI_FEATURE_COLUMNS],
    )

    # The cases every arm can be compared on. INSPECT does the same thing inline with
    # --compare_vs_pesi; writing it out lets any run restrict to the identical subset.
    evaluable_path = _write_csv(
        output_dir / "spesi_evaluable.csv",
        [
            {"patient_id": row["patient_id"], "study_id": row["study_id"], "split": row["split"]}
            for row in feature_rows
            if row["spesi_computable"]
        ],
        ["patient_id", "study_id", "split"],
    )

    scored = sum(row["spesi_computable"] for row in feature_rows)
    by_split: Counter[str] = Counter()
    distribution: Counter[int] = Counter()
    for record, row in zip(records, feature_rows):
        if row["spesi_computable"]:
            by_split[record.split] += 1
            distribution[int(row["spesi"])] += 1

    index = dict(mapping.get("index") or {})
    metadata: dict[str, Any] = {
        "status": "built" if scored else "built_no_complete_cases",
        "reason": (
            f"sPESI scored for {scored}/{len(records)} studies"
            if scored
            else "no study has all six sPESI components; see missing_component_cases"
        ),
        "score_specification": str(mapping.get("score_specification") or "sPESI"),
        "reference_implementation": str(mapping.get("reference_implementation") or ""),
        "mapping_config": str(mapping_path),
        "index": index,
        "index_overrides": overrides,
        "feature_table": str(features_path),
        "component_table": str(components_path),
        "evaluable_cohort": str(evaluable_path),
        "score_output_columns": list(SPESI_FEATURE_COLUMNS),
        "code_resolution": code_audit,
        "studies": len(records),
        "scored_cases": scored,
        "scored_cases_by_split": dict(sorted(by_split.items())),
        "score_distribution": dict(sorted(distribution.items())),
        "missing_component_cases": dict(sorted(missing_counts.items())),
        "skipped_events": dict(sorted(skipped.items())),
    }
    _write_json(output_dir / "spesi_status.json", metadata)
    _write_json(output_dir / "spesi_mapping_audit.json", {"mapping_config": str(mapping_path), "components": code_audit})
    return SpesiArtifacts(metadata["status"], SPESI_FEATURE_COLUMNS, by_study, metadata, components_by_study)
