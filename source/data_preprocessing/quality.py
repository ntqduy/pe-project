"""Compact, de-identified data-readiness reporting for derived INSPECT datasets.

The dataset build retains detailed, patient-linked failure records under ``audit/`` for
debugging.  This module produces the small top-level report people should read first:
cohort size, class distribution, data loss, cache health and clinical-data readiness.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adjudication import normalize_binary
from .sources import StudyRecord

SPLITS = ("train", "validation", "test")


def cohort_step(step: str, what: str, records: Sequence[StudyRecord]) -> dict[str, Any]:
    """One row of the cohort funnel: how many studies/patients survive a build step.

    The dataset build calls this after every step that can drop a case, so
    ``data_quality.md`` answers "how many samples are left after each step?" in one table
    instead of making the reader add up separate eligibility, integrity and cache reports.
    """
    return {
        "step": step,
        "what": what,
        "studies": len(records),
        "patients": len({record.patient_id for record in records}),
        "by_split": {
            split: sum(record.split == split for record in records) for split in SPLITS
        },
    }


def funnel_losses(funnel: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Annotate a funnel with the studies/patients each step removed."""
    annotated: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for entry in funnel:
        row = dict(entry)
        row["removed_studies"] = (
            0 if previous is None else int(previous["studies"]) - int(row["studies"])
        )
        row["removed_patients"] = (
            0 if previous is None else int(previous["patients"]) - int(row["patients"])
        )
        annotated.append(row)
        previous = row
    return annotated


def _split_counts(records: Sequence[StudyRecord]) -> dict[str, dict[str, int]]:
    patients: dict[str, set[str]] = defaultdict(set)
    studies: Counter[str] = Counter()
    for record in records:
        patients[record.split].add(record.patient_id)
        studies[record.split] += 1
    return {
        split: {"patients": len(patients[split]), "studies": int(studies[split])}
        for split in SPLITS
    }


def _label_distribution(
    records: Sequence[StudyRecord], columns: Sequence[str]
) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for column in columns:
        overall: Counter[str] = Counter()
        per_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
        for record in records:
            state = normalize_binary(record.labels.get(column))
            overall[state] += 1
            if record.split in per_split:
                per_split[record.split][state] += 1
        result[column] = {
            "all": dict(sorted(overall.items())),
            **{split: dict(sorted(per_split[split].items())) for split in SPLITS},
        }
    return result


def _crosswalk_coverage(records: Sequence[StudyRecord]) -> dict[str, Any]:
    total_studies = len(records)
    total_patients = len({record.patient_id for record in records})
    linked_studies = sum(record.has_ehr_crosswalk for record in records)
    linked_patients = len({record.patient_id for record in records if record.has_ehr_crosswalk})
    return {
        "linked_studies": linked_studies,
        "total_studies": total_studies,
        "study_coverage": linked_studies / total_studies if total_studies else None,
        "linked_patients": linked_patients,
        "total_patients": total_patients,
        "patient_coverage": linked_patients / total_patients if total_patients else None,
        "by_split": {
            split: {
                "linked_studies": sum(
                    record.has_ehr_crosswalk for record in records if record.split == split
                ),
                "total_studies": sum(record.split == split for record in records),
            }
            for split in SPLITS
        },
    }


def _number_bucket(value: Any, cuts: Sequence[float], labels: Sequence[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return "MISSING"
    try:
        number = float(text)
    except ValueError:
        return "INVALID"
    for cut, label in zip(cuts, labels):
        if number <= cut:
            return label
    return labels[-1]


def _top_categories(values: Sequence[str], *, limit: int = 8) -> dict[str, int]:
    counts = Counter(value.strip() or "MISSING" for value in values)
    ordered = counts.most_common(limit)
    result = dict(ordered)
    remaining = sum(count for _, count in counts.most_common()[limit:])
    if remaining:
        result["OTHER"] = remaining
    return result


def _ctpa_acquisition_summary(records: Sequence[StudyRecord]) -> dict[str, Any]:
    """Aggregate CT geometry and index-time availability without exposing case IDs."""
    return {
        "num_slices": dict(
            sorted(
                Counter(
                    _number_bucket(
                        record.series.get("num_slices"),
                        (1, 127, float("inf")),
                        ("<=1", "2-127", ">=128"),
                    )
                    for record in records
                ).items()
            )
        ),
        "slice_thickness_mm": dict(
            sorted(
                Counter(
                    _number_bucket(
                        record.series.get("SliceThickness"),
                        (1.5, 3.0, float("inf")),
                        ("<=1.5", "1.5-3", ">3"),
                    )
                    for record in records
                ).items()
            )
        ),
        "pixel_spacing_0_mm": dict(
            sorted(
                Counter(
                    _number_bucket(
                        record.series.get("PixelSpacing_0"),
                        (0.8, 1.0, float("inf")),
                        ("<=0.8", "0.8-1.0", ">1.0"),
                    )
                    for record in records
                ).items()
            )
        ),
        "manufacturer_top": _top_categories(
            [str(record.series.get("Manufacturer") or "") for record in records]
        ),
        "procedure_datetime": {
            "available": sum(bool(record.procedure_datetime.strip()) for record in records),
            "missing": sum(not record.procedure_datetime.strip() for record in records),
            "by_split": {
                split: {
                    "available": sum(
                        record.split == split and bool(record.procedure_datetime.strip())
                        for record in records
                    ),
                    "missing": sum(
                        record.split == split and not record.procedure_datetime.strip()
                        for record in records
                    ),
                }
                for split in SPLITS
            },
        },
    }


def _ehr_source_inventory(release_root: Path) -> dict[str, Any]:
    directory = release_root / "EHR"
    if not directory.is_dir():
        return {"status": "not_available", "directory": str(directory), "artifacts": []}
    artifacts = []
    for path in sorted(directory.iterdir()):
        if path.is_file():
            artifacts.append({"name": path.name, "bytes": path.stat().st_size})
    return {
        "status": "source_available" if artifacts else "empty",
        "directory": str(directory),
        "artifacts": artifacts,
    }


def build_data_quality(
    *,
    profile: Mapping[str, Any],
    release_root: Path,
    records: Sequence[StudyRecord],
    adjudication_columns: Sequence[str],
    eligibility: Mapping[str, Any],
    integrity: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    sampling: Mapping[str, Any],
    ehr: Mapping[str, Any] | None = None,
    pesi: Mapping[str, Any] | None = None,
    modality_availability: Mapping[str, Any] | None = None,
    cohort_funnel: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a compact, de-identified readiness report for one built profile."""
    cache_qc = dict(preprocessing.get("cache_qc") or {})
    ehr_build = dict(ehr or {})
    pesi_build = dict(pesi or {})
    return {
        "profile": dict(profile),
        "cohort": {
            "patients": len({record.patient_id for record in records}),
            "studies": len(records),
            "by_split": _split_counts(records),
        },
        "cohort_funnel": funnel_losses(cohort_funnel),
        "native_label_distribution": _label_distribution(records, adjudication_columns),
        "data_loss": {
            "eligibility": dict(eligibility),
            "integrity": dict(integrity),
            "preprocessing": {
                "enabled": preprocessing.get("enabled"),
                "written": preprocessing.get("written"),
                "cached": preprocessing.get("cached"),
                "failure_count": preprocessing.get("failure_count"),
                # The full failure list contains study IDs and stays in audit/cache_qc.json.
                "cache_qc": {
                    key: cache_qc.get(key)
                    for key in ("enabled", "checked", "passed", "failed", "skipped")
                },
            },
        },
        "clinical_readiness": {
            "ehr_crosswalk": _crosswalk_coverage(records),
            "ehr_source": _ehr_source_inventory(release_root),
            "ehr_build": ehr_build,
            "modality_availability": dict(modality_availability or {}),
            "ehr_temporal_boundary": (
                ehr_build.get("feature_end_operator")
                or "not_run: no EHR event feature table is built, so no pre-CTPA temporal audit exists"
            ),
            "pesi": pesi_build
            or {"status": "not_built", "reason": "no approved PESI/sPESI mapping is configured"},
        },
        "ctpa_acquisition": _ctpa_acquisition_summary(records),
        "sampling": dict(sampling),
        "split_audit": dict(split_audit),
    }


def _format_count_map(values: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items())) or "none"


def render_data_quality_markdown(payload: Mapping[str, Any]) -> str:
    """Render the human entry point without including patient, study or report IDs."""
    profile = dict(payload.get("profile") or {})
    cohort = dict(payload.get("cohort") or {})
    loss = dict(payload.get("data_loss") or {})
    clinical = dict(payload.get("clinical_readiness") or {})
    split_audit = dict(payload.get("split_audit") or {})
    sampling = dict(payload.get("sampling") or {})
    lines = [
        f"# Data quality — {profile.get('name', 'dataset')}",
        "",
        str(profile.get("description") or ""),
        "",
        "## Cohort",
        "",
        "| Split | Patients | Studies |",
        "| --- | ---: | ---: |",
    ]
    by_split = dict(cohort.get("by_split") or {})
    for split in SPLITS:
        row = dict(by_split.get(split) or {})
        lines.append(f"| {split} | {row.get('patients', 0)} | {row.get('studies', 0)} |")
    lines.append(f"| **All** | **{cohort.get('patients', 0)}** | **{cohort.get('studies', 0)}** |")

    funnel = list(payload.get("cohort_funnel") or [])
    if funnel:
        lines.extend(
            [
                "",
                "## Cohort funnel (samples remaining after each step)",
                "",
                "| Step | Studies | Patients | Studies removed | Patients removed | What it drops |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for entry in funnel:
            entry = dict(entry)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(entry.get("step", "")),
                        str(entry.get("studies", 0)),
                        str(entry.get("patients", 0)),
                        str(entry.get("removed_studies", 0)),
                        str(entry.get("removed_patients", 0)),
                        str(entry.get("what", "")),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Native-label distribution (studies)", ""])
    distributions = dict(payload.get("native_label_distribution") or {})
    if distributions:
        lines.extend(["| Label | All | Train | Validation | Test |", "| --- | --- | --- | --- | --- |"])
        for label, values in sorted(distributions.items()):
            values = dict(values)
            lines.append(
                "| " + " | ".join(
                    [
                        str(label),
                        _format_count_map(dict(values.get("all") or {})),
                        _format_count_map(dict(values.get("train") or {})),
                        _format_count_map(dict(values.get("validation") or {})),
                        _format_count_map(dict(values.get("test") or {})),
                    ]
                ) + " |"
            )
    else:
        lines.append("No native-label columns were configured.")

    acquisition = dict(payload.get("ctpa_acquisition") or {})
    procedure_datetime = dict(acquisition.get("procedure_datetime") or {})
    lines.extend(
        [
            "",
            "## CTPA acquisition and index-time readiness",
            "",
            "| Attribute | Distribution |",
            "| --- | --- |",
            "| Slices | " + _format_count_map(dict(acquisition.get("num_slices") or {})) + " |",
            "| Slice thickness (mm) | "
            + _format_count_map(dict(acquisition.get("slice_thickness_mm") or {}))
            + " |",
            "| Pixel spacing 0 (mm) | "
            + _format_count_map(dict(acquisition.get("pixel_spacing_0_mm") or {}))
            + " |",
            "| Manufacturer (top 8) | "
            + _format_count_map(dict(acquisition.get("manufacturer_top") or {}))
            + " |",
            f"| CTPA procedure time | available={procedure_datetime.get('available', 0)}, "
            f"missing={procedure_datetime.get('missing', 0)} |",
        ]
    )

    eligibility = dict(loss.get("eligibility") or {})
    integrity = dict(loss.get("integrity") or {})
    preprocessing = dict(loss.get("preprocessing") or {})
    lines.extend(
        [
            "",
            "## QC and data loss",
            "",
            f"- Eligibility: {eligibility.get('considered_studies', 0)} considered; "
            f"{eligibility.get('eligible_studies', 0)} eligible; "
            f"{eligibility.get('excluded_studies', 0)} excluded.",
            "- Exclusions by rule: " + _format_count_map(dict(eligibility.get("excluded_by_rule") or {})) + ".",
            f"- CT integrity: {integrity.get('checked', 0)} checked; "
            + _format_count_map(dict(integrity.get("by_status") or {}))
            + ".",
            f"- CT cache: written={preprocessing.get('written', 0)}, "
            f"cached={preprocessing.get('cached', 0)}, "
            f"preprocessing failures={preprocessing.get('failure_count', 0)}.",
            "- Cache QC: " + _format_count_map(dict(preprocessing.get("cache_qc") or {})) + ".",
            "",
            "## Clinical readiness",
            "",
        ]
    )
    crosswalk = dict(clinical.get("ehr_crosswalk") or {})
    lines.append(
        f"- EHR crosswalk: {crosswalk.get('linked_studies', 0)}/{crosswalk.get('total_studies', 0)} "
        f"studies and {crosswalk.get('linked_patients', 0)}/{crosswalk.get('total_patients', 0)} patients linked."
    )
    ehr_source = dict(clinical.get("ehr_source") or {})
    lines.append(
        f"- EHR source: {ehr_source.get('status', 'unknown')}."
    )
    lines.append(
        "- EHR temporal boundary: "
        + str(clinical.get("ehr_temporal_boundary", "not configured"))
        + "."
    )
    ehr_build = dict(clinical.get("ehr_build") or {})
    if ehr_build:
        lines.append(
            f"- EHR build: {ehr_build.get('status', 'unknown')}; "
            f"cases={ehr_build.get('cases', 0)}, "
            f"missing index time={ehr_build.get('missing_index_time', 0)}, "
            f"EHR-missing cases={ehr_build.get('ehr_missing_cases', 0)}."
        )
        lines.append(
            "- EHR excluded case-events: "
            + _format_count_map(dict(ehr_build.get("excluded_case_event_counts") or {}))
            + "."
        )
        profiles = dict(ehr_build.get("profiles") or {})
        if profiles:
            lines.append("- EHR input profiles (all strictly pre-CTPA):")
            for name, profile in profiles.items():
                profile = dict(profile or {})
                lines.append(
                    f"  - {name}: cutoff={profile.get('end_before_ctpa_hours', 'unknown')}h; "
                    f"EHR-missing cases={profile.get('ehr_missing_cases', 'unknown')}; "
                    f"columns={len(profile.get('manifest_feature_columns') or ())}."
                )
    pesi = dict(clinical.get("pesi") or {})
    lines.append(
        f"- PESI/sPESI: {pesi.get('status', 'not configured')}; "
        f"{pesi.get('reason', 'no status reason')}"
    )
    candidate_coverage = dict(pesi.get("candidate_coverage") or {})
    candidate_components = dict(candidate_coverage.get("components") or {})
    if candidate_components:
        candidate_summary = ", ".join(
            f"{name}={dict(candidate_components.get(name) or {}).get('cases_with_eligible_candidate_event', 0)}"
            for name in ("pulse", "respiratory_rate", "temperature_c", "oxygen_saturation")
        )
        lines.append(
            "- PESI candidate pre-CTPA availability (cases; audit only): "
            + candidate_summary
            + "."
        )
    candidate_table = dict(pesi.get("candidate_components") or {})
    if candidate_table.get("component_table"):
        lines.append(
            "- Raw PESI-component candidates: "
            f"any={candidate_table.get('cases_with_any_component', 0)}, "
            f"all 11={candidate_table.get('cases_with_all_components', 0)}; "
            "the table is unscored and retains per-feature missingness."
        )
    modality = dict(clinical.get("modality_availability") or {})
    availability = dict(modality.get("coverage") or {})
    if availability:
        lines.append(
            "- Modality availability: "
            + ", ".join(
                f"{name}={dict(values).get('available_cases', 0)}/"
                f"{dict(values).get('total_cases', 0)}"
                for name, values in availability.items()
            )
            + "."
        )
    lines.extend(
        [
            "",
            "## Sampling and leakage guard",
            "",
            f"- Sampling: {'enabled' if sampling.get('enabled', False) else 'not sampled'}.",
            f"- Split audit: {'PASS' if split_audit.get('ok') else 'FAIL'}; "
            + _format_count_map(dict(split_audit.get("split_patients") or {}))
            + " patients by split.",
            "",
            "Detailed patient-linked failures, if any, are in `audit/`; this report deliberately "
            "contains aggregate counts only.",
            "",
        ]
    )
    return "\n".join(lines)
