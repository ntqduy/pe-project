"""PNG figures for the EDA run.

Every figure degrades to a recorded skip instead of raising: a missing plotting backend
must not cost you the whole numeric report.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(figure, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120, bbox_inches="tight")
    return str(destination.resolve())


def split_composition(cohort: Mapping[str, Any], destination: Path) -> str | None:
    manifests = cohort.get("per_manifest") or {}
    source = manifests.get("ctpa.csv") or next(iter(manifests.values()), None)
    if not source or not source.get("splits"):
        return None
    plt = _pyplot()
    splits = source["splits"]
    names = list(splits)
    patients = [splits[name]["patients"] for name in names]
    studies = [splits[name]["studies"] for name in names]
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    positions = range(len(names))
    axis.bar([p - 0.2 for p in positions], patients, width=0.4, label="patients")
    axis.bar([p + 0.2 for p in positions], studies, width=0.4, label="studies")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(names)
    axis.set_ylabel("count")
    axis.set_title("Split composition")
    axis.legend()
    path = _save(figure, destination)
    plt.close(figure)
    return path


def label_prevalence(labels: Mapping[str, Any], destination: Path) -> str | None:
    overall = labels.get("overall") or {}
    usable = {
        name: values["prevalence_of_observed"]
        for name, values in overall.items()
        if values.get("prevalence_of_observed") is not None
    }
    if not usable:
        return None
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    names = list(usable)
    axis.bar(names, [usable[name] for name in names], color="steelblue")
    axis.set_ylabel("prevalence of observed rows")
    axis.set_title("Diagnosis label prevalence (missing excluded)")
    axis.tick_params(axis="x", rotation=30)
    path = _save(figure, destination)
    plt.close(figure)
    return path


def label_missingness(labels: Mapping[str, Any], destination: Path) -> str | None:
    overall = labels.get("overall") or {}
    usable = {
        name: values["missing_rate"]
        for name, values in overall.items()
        if values.get("missing_rate") is not None
    }
    if not usable:
        return None
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    names = list(usable)
    axis.bar(names, [usable[name] for name in names], color="indianred")
    axis.set_ylabel("missing rate")
    axis.set_title("Diagnosis label missingness")
    axis.tick_params(axis="x", rotation=30)
    path = _save(figure, destination)
    plt.close(figure)
    return path


def endpoint_events(outcomes: Mapping[str, Any], destination: Path) -> str | None:
    for name, payload in outcomes.items():
        if not isinstance(payload, Mapping) or payload.get("status") != "available":
            continue
        endpoints = payload.get("endpoints") or {}
        if not endpoints:
            continue
        plt = _pyplot()
        names = list(endpoints)
        events = [endpoints[key]["events"] for key in names]
        figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        axis.bar(names, events, color="seagreen")
        axis.axhline(100, linestyle="--", linewidth=1, color="grey",
                     label="100 events (single hold-out guidance)")
        axis.set_ylabel("outcome events")
        axis.set_title(f"Prognosis events per endpoint ({name})")
        axis.tick_params(axis="x", rotation=30)
        axis.legend()
        path = _save(figure, destination)
        plt.close(figure)
        return path
    return None


def report_length(reports: Mapping[str, Any], destination: Path) -> str | None:
    text = reports.get("text") or {}
    words = text.get("words")
    if not words:
        return None
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    keys = ["min", "p25", "median", "p75", "max"]
    axis.bar(keys, [words[key] for key in keys], color="darkorange")
    axis.set_ylabel("words per report")
    axis.set_title("Report length distribution (quantiles)")
    path = _save(figure, destination)
    plt.close(figure)
    return path


def render_all(summary: Mapping[str, Any], figures_dir: Path) -> dict[str, Any]:
    """Render every figure, recording per-figure skips rather than aborting."""
    produced: dict[str, Any] = {}
    plan = (
        ("split_composition", split_composition, "cohort"),
        ("label_prevalence", label_prevalence, "labels"),
        ("label_missingness", label_missingness, "labels"),
        ("endpoint_events", endpoint_events, "outcomes"),
        ("report_length", report_length, "reports"),
    )
    for name, function, section in plan:
        payload = summary.get(section) or {}
        try:
            path = function(payload, figures_dir / f"{name}.png")
        except Exception as exc:  # noqa: BLE001 - a plotting backend problem is not fatal
            produced[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            continue
        produced[name] = (
            {"status": "written", "path": path}
            if path
            else {"status": "skipped", "reason": "input section had nothing to plot"}
        )
    return produced
