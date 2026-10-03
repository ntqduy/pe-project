#!/usr/bin/env python
"""Collect the test results of one baseline experiment into tables and plots.

    python tools/baselines/summarize.py --exp exp01_baselines [--profile full_inspect]

Reads every ``runs/<model>__<head>__frac<PPP>[__v<variant>][__x<settings>]/official_seed<S>/
epoch_<E>/result.csv`` the experiment needs (tools/baselines/experiments.py) and writes, into
``<outputs>/<family>/BASE/<profile>/<task>/<exp>/``:

    runs.csv      one row per finished run (test metrics, validation AUROC, weights, path)
    summary.csv   mean / std / n over seeds per (model, head, fraction, variant)
    summary.md    the same as a readable table (+ which runs are still missing)
    *.png         exp01: test AUROC per model | exp02: AUROC vs training fraction |
                  exp03: MLP vs KAN per model
Only the test split is summarised; thresholds come from validation (evaluate.py).

Runs trained with non-default settings (``__x<settings>``: --scratch, --lr, --batch-size, ...)
are kept apart: they get their own rows, labelled in the ``settings`` column, and are never
pooled with the default runs. --settings default|<stamp> restricts the summary to one of them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.baselines.experiments import (  # noqa: E402
    DEFAULT_VARIANT,
    EXPERIMENTS,
    MODEL_GROUPS,
    PROGNOSIS_LABELS,
    base_directory,
    check_label,
    model_dimension,
    task_directory,
)

METRICS = ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy", "accuracy",
           "brier")
# Test-row columns of each run's result.csv copied into summary_raw.csv (source/metrics/result_table.py).
RESULT_COLUMNS = ("auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high", "threshold",
                  "threshold_rule", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy", "accuracy",
                  "brier", "note")
TAG = re.compile(
    r"^(?P<model>.+?)__(?P<head>mlp|kan)__frac(?P<fraction>\d{3})"
    r"(?:__v(?P<variant>[A-Za-z0-9_]+?))?(?:__x(?P<settings>[A-Za-z0-9._+-]+))?$"
)
RUN = re.compile(r"^official_seed(?P<seed>\d+)$")

# Reference categorical palette (light surface), fixed order - see the dataviz skill.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUP_COLORS = {"CNN": SERIES[0], "Transformer": SERIES[1], "Mamba": SERIES[2], "PE-specific": SERIES[3],
                "PE-specific / FM": SERIES[3]}


def _float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def collect(base: Path, spec: dict, exp: str, settings_filter: str | None = None) -> list[dict]:
    """Finished runs of ``exp``; settings_filter None = every settings stamp, "" = defaults only."""
    rows = []
    # One run folder may hold several epoch_<E> bundles (re-runs with another budget): the
    # most recently evaluated one represents the run, so a case is never counted twice.
    latest: dict[Path, Path] = {}
    for result in (base / "runs").glob("*/*/epoch_*/result.csv"):
        run_dir = result.parent.parent
        if run_dir not in latest or result.stat().st_mtime > latest[run_dir].stat().st_mtime:
            latest[run_dir] = result
    for result in sorted(latest.values()):
        epoch_dir = result.parent
        tag, run = TAG.match(epoch_dir.parent.parent.name), RUN.match(epoch_dir.parent.name)
        if not tag or not run:
            continue
        model, head, fraction = tag["model"], tag["head"], int(tag["fraction"])
        if model not in spec["models"] or head not in spec["heads"] or fraction not in spec["fractions"]:
            continue
        # A model with overrides in this experiment is represented by its own variant runs only;
        # an experiment with named variants (exp04) takes those plus the shared default run.
        variant = tag["variant"] or None
        if spec["overrides"].get(model):
            if variant != exp:
                continue
        elif (variant or DEFAULT_VARIANT) not in spec["variants"]:
            continue
        settings = tag["settings"] or ""
        if settings_filter is not None and settings != settings_filter:
            continue
        with result.open(newline="", encoding="utf-8") as handle:
            table = list(csv.DictReader(handle))
        weights, params, primary = "", "", None
        try:
            payload = json.loads((epoch_dir / "result.json").read_text(encoding="utf-8"))
            model_block = payload.get("model") or {}
            weights = str((model_block.get("pretrained_weights") or {}).get("status") or "")
            count = model_block.get("parameters") or model_block.get("total_params")
            params = f"{float(count) / 1e6:.1f}" if count else ""
            primary = (payload.get("evaluation") or {}).get("primary_target") or (
                (payload.get("config") or {}).get("task") or {}).get("primary_target")
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        # result.csv has one row per (target, split); a multitask run adds auxiliary targets,
        # so only the run's primary target may stand for it.
        if primary is not None:
            table = [row for row in table if row.get("target") in (None, "", str(primary))]
        elif len({row.get("target") for row in table}) > 1:
            print(f"warning: {epoch_dir}: several targets in result.csv and no primary_target in "
                  "result.json; skipped", file=sys.stderr)
            continue
        test = next((row for row in table if row.get("split") == "test"), None)
        validation = next((row for row in table if row.get("split") == "validation"), None)
        if test is None:
            continue
        rows.append({
            "model": model, "dim": model_dimension(model), "group": MODEL_GROUPS.get(model, ""), "head": head,
            "fraction": fraction, "variant": variant or DEFAULT_VARIANT, "settings": settings,
            "seed": int(run["seed"]),
            "epochs": epoch_dir.name,
            "n_test": test.get("n_studies"),
            **{column: (test.get(column) or "") if column in ("threshold_rule", "note") else _float(test.get(column))
               for column in RESULT_COLUMNS},
            "val_auroc": _float((validation or {}).get("auroc")), "weights": weights, "params_M": params,
            "primary_target": primary or "", "run_dir": str(epoch_dir),
        })
    return rows


def group_key(row: dict) -> tuple:
    """Runs pooled into one summary row: same model, head, fraction, variant and settings."""
    return row["model"], row["head"], row["fraction"], row["variant"], row["settings"]


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    summary = []
    for (model, head, fraction, variant, settings), items in groups.items():
        entry = {"model": model, "dim": model_dimension(model), "group": MODEL_GROUPS.get(model, ""), "head": head,
                 "fraction": fraction, "variant": variant, "settings": settings, "n_runs": len(items),
                 "n_seeds": len({item["seed"] for item in items}),
                 "seeds": " ".join(str(seed) for seed in sorted({item["seed"] for item in items})),
                 "weights": ",".join(sorted({item["weights"] for item in items}))}
        for metric in METRICS + ("val_auroc",):
            values = [item[metric] for item in items if math.isfinite(item[metric])]
            entry[f"{metric}_mean"] = statistics.fmean(values) if values else float("nan")
            entry[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else float("nan")
        if len(items) == 1:
            entry["auroc_ci"] = f"[{items[0]['auroc_ci_low']:.3f}, {items[0]['auroc_ci_high']:.3f}]"
        summary.append(entry)
    order = {model: index for index, model in enumerate(sum((EXPERIMENTS[e]["models"] for e in EXPERIMENTS), []))}
    return sorted(summary, key=lambda item: (order.get(item["model"], 99), item["head"],
                                             item["fraction"], item["variant"] != DEFAULT_VARIANT, item["variant"],
                                             item["settings"]))


def _merged_predictions(items: list[dict]) -> dict[str, list[dict]]:
    """Seed-averaged probabilities per split, merged on study_id (never on row order).

    Every run of the group must hold the same studies with the same patient and label in
    each split; any difference is an error, not something to skip silently.
    """
    from source.metrics.result_table import read_prediction_file

    merged: dict[str, dict[str, dict]] = {}
    reference_run: dict[str, str] = {}
    for item in items:
        path = Path(item["run_dir"]) / "predictions.csv"
        if not path.is_file():
            raise SystemExit(f"seed ensemble: {path} is missing")
        primary = item.get("primary_target") or ""
        rows = [row for row in read_prediction_file(path, split=None)
                if not primary or str(row.get("target")) == primary]
        by_split: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            study = str(row["study_id"])
            if study in by_split[row["split"]]:
                raise SystemExit(f"seed ensemble: study_id {study} appears twice in {path} ({row['split']})")
            by_split[row["split"]][study] = row
        for split, studies in by_split.items():
            if split not in merged:
                merged[split] = {study: {"patient_id": str(row["patient_id"]), "study_id": study,
                                         "y_true": int(row["y_true"]), "probabilities": [float(row["y_prob"])]}
                                 for study, row in studies.items()}
                reference_run[split] = str(path)
                continue
            expected, found = set(merged[split]), set(studies)
            if expected != found:
                raise SystemExit(
                    f"seed ensemble: {split} studies differ between {reference_run[split]} and {path} "
                    f"(only in first: {sorted(expected - found)[:5]}, only in second: {sorted(found - expected)[:5]})")
            for study, row in studies.items():
                target = merged[split][study]
                if int(row["y_true"]) != target["y_true"] or str(row["patient_id"]) != target["patient_id"]:
                    raise SystemExit(f"seed ensemble: study {study} ({split}) has another label or patient in {path}")
                target["probabilities"].append(float(row["y_prob"]))
    return {split: [{"patient_id": row["patient_id"], "study_id": row["study_id"], "y_true": row["y_true"],
                     "y_prob": statistics.fmean(row["probabilities"])} for row in studies.values()]
            for split, studies in merged.items()}


def seed_ensemble(items: list[dict], stage: str) -> dict:
    """result.csv-style test row of the seed-averaged prediction of one group of runs.

    Threshold: Youden (or the runs' rule) on the averaged validation probabilities, applied
    unchanged to test; 95% CI: patient-level bootstrap of the averaged test probabilities.
    """
    from source.metrics.result_table import metric_bundle, result_rows, select_threshold, split_summary

    method, samples, confidence = "youden", 2000, 0.95
    try:
        payload = json.loads((Path(items[0]["run_dir"]) / "result.json").read_text(encoding="utf-8"))
        evaluation = payload.get("evaluation") or {}
        method = str(evaluation.get("threshold_method") or method)
        samples = int((evaluation.get("bootstrap") or {}).get("samples") or samples)
        confidence = float((evaluation.get("bootstrap") or {}).get("confidence") or confidence)
    except (OSError, ValueError, TypeError):
        pass
    splits = _merged_predictions(items)
    test = splits.get("test") or []
    validation = splits.get("validation") or []
    if not test:
        raise SystemExit(f"seed ensemble: no test predictions in {items[0]['run_dir']}")
    threshold, source = select_threshold(validation, method) if validation else (0.5, "")
    metrics, reason = metric_bundle(test, stage, threshold, with_ci=True, samples=samples, confidence=confidence, seed=42)
    row = result_rows(
        experiment="seed_ensemble", target=str(items[0].get("primary_target") or ""),
        splits={"test": {"metrics": metrics, "summary": split_summary(test, threshold), "ci_reason": reason}},
        threshold=threshold, threshold_source=source, stage=stage, threshold_method=method,
    )[0]
    return {**row, "n_seeds": len({item["seed"] for item in items}), "bootstrap_samples": samples}


def _interval(value, low, high) -> str:
    value, low, high = _float(value), _float(low), _float(high)
    if not math.isfinite(value):
        return "-"
    if not (math.isfinite(low) and math.isfinite(high)):
        return f"{value:.3f}"
    return f"{value:.3f} [{low:.3f}–{high:.3f}]"


def _cell(entry: dict, metric: str) -> str:
    mean, std = entry[f"{metric}_mean"], entry[f"{metric}_std"]
    if not math.isfinite(mean):
        return "-"
    return f"{mean:.3f}" + (f" ± {std:.3f}" if math.isfinite(std) else "")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


PRETTY_POINT_METRICS = ("sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy", "accuracy", "brier")


def pretty_rows(summary: list[dict], ensembles: dict[tuple, dict]) -> list[dict]:
    """One readable row per (model, head, fraction, variant, settings).

    AUROC / AUPRC: value [95% CI] of the seed-averaged prediction (seed_ensemble) next to
    mean ± std of the per-seed values; every other metric: mean ± std over the seeds.
    """
    rows = []
    for entry in summary:
        ensemble = ensembles.get(group_key(entry)) or {}
        row = {"model": entry["model"], "dim": entry["dim"], "group": entry["group"], "head": entry["head"],
               "train_%": entry["fraction"], "variant": entry["variant"], "settings": entry["settings"] or "default",
               "n_seeds": entry["n_seeds"], "seeds": entry["seeds"]}
        for metric in ("auroc", "auprc"):
            row[f"{metric}_[CI]_seed_avg"] = _interval(ensemble.get(metric), ensemble.get(f"{metric}_ci_low"),
                                                       ensemble.get(f"{metric}_ci_high"))
            row[f"{metric}_mean±std"] = _cell(entry, metric)
        row["threshold_seed_avg"] = "-" if not math.isfinite(_float(ensemble.get("threshold"))) else f"{_float(ensemble.get('threshold')):.3f}"
        row["threshold_rule"] = ensemble.get("threshold_rule", "")
        for metric in PRETTY_POINT_METRICS:
            row[f"{metric}_mean±std"] = _cell(entry, metric)
        row["val_auroc_mean±std"] = _cell(entry, "val_auroc")
        row["weights"] = entry["weights"]
        row["note"] = ensemble.get("note", "")
        rows.append(row)
    return rows


def write_tables(out: Path, exp: str, spec: dict, rows: list[dict], summary: list[dict],
                 ensembles: dict[tuple, dict]) -> None:
    """summary_raw.csv (every run), summary.csv (numeric mean/std), summary_ensemble.csv
    (seed-averaged prediction, result.csv columns), summary_pretty.csv + summary.md."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.csv").unlink(missing_ok=True)       # renamed summary_raw.csv
    if rows:
        _write_csv(out / "summary_raw.csv", rows)
    if summary:
        _write_csv(out / "summary.csv", summary)
    if ensembles:
        _write_csv(out / "summary_ensemble.csv", [
            {"model": key[0], "head": key[1], "fraction": key[2], "variant": key[3], "settings": key[4] or "default",
             **value} for key, value in ensembles.items()])
    pretty = pretty_rows(summary, ensembles)
    if pretty:
        _write_csv(out / "summary_pretty.csv", pretty)
    done = {(entry["model"], entry["head"], entry["fraction"], entry["variant"]) for entry in summary if not entry["settings"]}
    missing = [f"{m} / {h} / {f}%" + (f" / {v}" if v != DEFAULT_VARIANT else "")
               for m in spec["models"] for h in spec["heads"] for f in spec["fractions"] for v in spec["variants"]
               if (m, h, f, v) not in done]
    columns = ["model", "dim", "head", "train_%", "variant", "settings", "n_seeds", "auroc_[CI]_seed_avg", "auroc_mean±std",
               "auprc_[CI]_seed_avg", "auprc_mean±std", "sensitivity_mean±std", "specificity_mean±std",
               "f1_mean±std", "balanced_accuracy_mean±std", "brier_mean±std", "weights"]
    titles = ["model", "dim", "head", "train %", "variant", "settings", "n seeds", "AUROC [95% CI]", "AUROC mean ± std",
              "AUPRC [95% CI]", "AUPRC mean ± std", "Sens", "Spec", "F1", "Bal.Acc", "Brier", "weights"]
    lines = [f"# {exp}: {spec['title']}", "",
             "Test split (official INSPECT test), threshold chosen on validation. "
             "`[95% CI]`: patient-bootstrap interval of the seed-averaged prediction (probabilities "
             "averaged per study_id over the seeds, threshold re-selected on the averaged validation "
             "predictions); `mean ± std`: over the seeds. Full columns: summary_pretty.csv, "
             "summary_ensemble.csv, summary_raw.csv.", "",
             "| " + " | ".join(titles) + " |", "|" + "---|" * len(titles)]
    for row in pretty:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if missing:
        lines += ["", f"Not finished yet ({len(missing)}): " + ", ".join(missing)]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def link_runs(out: Path, exp: str, rows: list[dict]) -> int:
    """<exp>/runs/<model>_<head>[_frac<PPP>]/official_seed<S> -> the shared run folder.

    Runs are stored once (shared by experiments); these links give every experiment the
    per-experiment layout of the proposal. Where symlinks are not allowed (Windows without
    developer mode) a LINK.txt with the target path is written instead.
    """
    import os

    made = 0
    for row in rows:
        name = (f"{row['model']}_{row['head']}" + (f"_frac{int(row['fraction']):03d}" if int(row["fraction"]) != 100 else "")
                + (f"_v{row['variant']}" if row["variant"] != DEFAULT_VARIANT else "")
                + (f"_x{row['settings']}" if row["settings"] else ""))
        target = Path(row["run_dir"]).parent             # official_seed<S> (all epoch bundles)
        link = out / "runs" / name / target.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            continue
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError:
            link.mkdir(parents=True, exist_ok=True)
            (link / "LINK.txt").write_text(str(target) + "\n", encoding="utf-8")
        made += 1
    return made


def _axes(title: str, width: float = 10.0, height: float = 4.8):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(width, height), dpi=150)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_2, labelsize=8)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_title(title, color=INK, fontsize=11, loc="left")
    return plt, figure, axis


def _series(entry: dict) -> str:
    """Plot label: the model, plus its settings stamp when it is not a default run."""
    return entry["model"] + (f" [{entry['settings']}]" if entry["settings"] else "")


def plot(out: Path, exp: str, summary: list[dict]) -> list[Path]:
    if not summary:
        return []
    written = []
    if exp == "exp01_baselines":
        plt, figure, axis = _axes("Test AUROC per baseline (mean ± std over seeds)", 12, 5)
        names = [entry["model"] + (f" [{entry['settings']}]" if entry["settings"] else "") for entry in summary]
        values = [entry["auroc_mean"] for entry in summary]
        errors = [entry["auroc_std"] if math.isfinite(entry["auroc_std"]) else 0 for entry in summary]
        colors = [GROUP_COLORS.get(entry["group"], SERIES[6]) for entry in summary]
        bars = axis.bar(range(len(names)), values, yerr=errors, color=colors, width=0.7, edgecolor=SURFACE,
                        linewidth=2, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        axis.set_xticks(range(len(names)), names, rotation=60, ha="right")
        axis.axhline(0.5, color=INK_2, linewidth=0.8, linestyle="--")
        axis.set_ylabel("test AUROC", color=INK_2, fontsize=9)
        axis.set_ylim(0.4, 1.0)
        for bar, value in zip(bars, values):
            if math.isfinite(value):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=6, color=INK_2)
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (SERIES[0], SERIES[1], SERIES[2], SERIES[3])]
        axis.legend(handles, ["CNN", "Transformer", "Mamba", "PE-specific / FM"], frameon=False, fontsize=8, ncol=4, loc="upper left")
        figure.tight_layout()
        written.append(out / "auroc_per_model.png")
    elif exp == "exp02_data_fraction":
        plt, figure, axis = _axes("Test AUROC vs share of training patients (mean ± std)")
        models = list(dict.fromkeys(_series(entry) for entry in summary))
        for index, model in enumerate(models):
            points = sorted((entry for entry in summary if _series(entry) == model), key=lambda entry: entry["fraction"])
            xs = [entry["fraction"] for entry in points]
            ys = [entry["auroc_mean"] for entry in points]
            es = [entry["auroc_std"] if math.isfinite(entry["auroc_std"]) else 0 for entry in points]
            color = SERIES[index % len(SERIES)]
            axis.errorbar(xs, ys, yerr=es, color=color, linewidth=2, marker="o", markersize=5, capsize=2, label=model,
                          markeredgecolor=SURFACE, markeredgewidth=1.5)
            if xs:
                axis.annotate(model, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=7,
                              color=INK_2, va="center")
        axis.set_xticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"])
        axis.set_xlim(20, 118)
        axis.set_xlabel("training patients used", color=INK_2, fontsize=9)
        axis.set_ylabel("test AUROC", color=INK_2, fontsize=9)
        axis.legend(frameon=False, fontsize=7, ncol=4, loc="lower right")
        figure.tight_layout()
        written.append(out / "auroc_vs_fraction.png")
    elif exp == "exp04_slice_ablation":
        plt, figure, axis = _axes("Test AUROC: slice selection / MIL pooling (mean ± std)")
        models = list(dict.fromkeys(_series(entry) for entry in summary))
        variants = list(dict.fromkeys(entry["variant"] for entry in summary))
        width = 0.8 / max(1, len(variants))
        for offset, variant in enumerate(variants):
            values, errors = [], []
            for model in models:
                entry = next((e for e in summary if _series(e) == model and e["variant"] == variant), None)
                values.append(entry["auroc_mean"] if entry else float("nan"))
                errors.append(entry["auroc_std"] if entry and math.isfinite(entry["auroc_std"]) else 0)
            positions = [index + (offset - (len(variants) - 1) / 2) * width for index in range(len(models))]
            label = "attention MIL (default)" if variant == DEFAULT_VARIANT else variant
            axis.bar(positions, values, width=width * 0.95, yerr=errors, color=SERIES[offset % len(SERIES)], label=label,
                     edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        axis.set_xticks(range(len(models)), models)
        axis.set_ylim(0.4, 1.0)
        axis.axhline(0.5, color=INK_2, linewidth=0.8, linestyle="--")
        axis.set_ylabel("test AUROC", color=INK_2, fontsize=9)
        axis.legend(frameon=False, fontsize=8, loc="upper left")
        figure.tight_layout()
        written.append(out / "slice_ablation.png")
    elif exp == "exp03_head_ablation":
        plt, figure, axis = _axes("Test AUROC: MLP vs KAN head (mean ± std)")
        models = list(dict.fromkeys(_series(entry) for entry in summary))
        for offset, (head, color) in enumerate((("mlp", SERIES[0]), ("kan", SERIES[1]))):
            values, errors = [], []
            for model in models:
                entry = next((e for e in summary if _series(e) == model and e["head"] == head), None)
                values.append(entry["auroc_mean"] if entry else float("nan"))
                errors.append(entry["auroc_std"] if entry and math.isfinite(entry["auroc_std"]) else 0)
            positions = [index + (offset - 0.5) * 0.38 for index in range(len(models))]
            axis.bar(positions, values, width=0.36, yerr=errors, color=color, label=head.upper(), edgecolor=SURFACE,
                     linewidth=2, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        axis.set_xticks(range(len(models)), models, rotation=30, ha="right")
        axis.set_ylim(0.4, 1.0)
        axis.axhline(0.5, color=INK_2, linewidth=0.8, linestyle="--")
        axis.set_ylabel("test AUROC", color=INK_2, fontsize=9)
        axis.legend(frameon=False, fontsize=8, loc="upper left")
        figure.tight_layout()
        written.append(out / "mlp_vs_kan.png")
    if written:
        figure.savefig(written[0], facecolor=SURFACE)
        plt.close("all")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--profile", default="full_inspect")
    parser.add_argument("--task", default="diagnosis", choices=["diagnosis", "prognosis"])
    parser.add_argument("--label", default=None, choices=PROGNOSIS_LABELS,
                        help="prognosis endpoint; required with --task prognosis, not allowed with diagnosis")
    parser.add_argument("--cohort", default="all")
    parser.add_argument("--base", type=Path, help="override <outputs>/<family>/BASE/<profile>/<task>")
    parser.add_argument("--settings", default=None,
                        help="only runs with this settings stamp ('default' = no __x suffix); default: all, kept apart")
    args = parser.parse_args(argv)
    check_label(args.task, args.label)
    spec = EXPERIMENTS[args.exp]
    base = args.base or base_directory(args.profile, task_directory(args.task, args.cohort, args.label))
    settings_filter = None if args.settings is None else ("" if args.settings == "default" else args.settings)
    rows = collect(base, spec, args.exp, settings_filter)
    summary = aggregate(rows)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    ensembles = {key: seed_ensemble(items, args.task) for key, items in groups.items()}
    out = base / args.exp
    write_tables(out, args.exp, spec, rows, summary, ensembles)
    link_runs(out, args.exp, rows)
    figures = plot(out, args.exp, summary)
    print(f"==> {args.exp}: {len(rows)} run(s), {len(summary)} row(s) -> {out}")
    for path in [out / "summary.md", out / "summary_pretty.csv", out / "summary_ensemble.csv", out / "summary.csv",
                 out / "summary_raw.csv", *figures]:
        if path.exists():
            print(f"    {path}")
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
