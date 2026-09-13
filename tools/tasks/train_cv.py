"""Patient-level stratified K-fold cross-validation over an existing task config.

The study protocol makes patient-level stratified three-fold CV the primary evaluation when
a reliable temporal hold-out is not available. This runner does not replace the trainer: it
materialises one manifest per fold with the ``split`` column rewritten, then invokes the
unchanged ``train_task.py`` and ``evaluate.py`` once per fold and aggregates the results.

    python tools/tasks/train_cv.py --config configs/runs/03_diagnosis/matrix/single_task.yaml --gpus 0

Fold construction, per outer fold k:

    test        = patients whose outer stratified fold is k
    validation  = one inner stratified fold carved from the remaining patients
    train       = the rest

Stratification is on the primary target and is always at patient level, so a patient can
never appear in two of the three roles. When the target cannot be stratified (missing
column, or a class with fewer members than k) the runner falls back to unstratified
patient K-fold and records why in the summary rather than silently changing the design.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.cv import (
    CrossValidationError,
    patient_kfold_assignments,
    stratified_patient_kfold_assignments,
)
from source.data.manifests import read_rows
from source.data.paths import ProjectPaths
from source.utils.config import load_config
from source.utils.logger import RunLogger
from tools._common import resolve_manifest

ROOT = Path(__file__).resolve().parents[2]


def _stratification_target(config: Mapping[str, Any], columns: set[str]) -> str | None:
    """The column folds are balanced on: primary target first, then the first label column."""
    task = dict(config.get("task") or {})
    data = dict(config.get("data") or {})
    candidates = [str(task.get("primary_target") or "")]
    candidates += [str(name) for name in (data.get("label_columns") or ())]
    return next((name for name in candidates if name and name in columns), None)


def _assign(
    rows: Sequence[Mapping[str, Any]], folds: int, seed: int, target: str | None
) -> tuple[dict[str, int], str]:
    """Stratified assignment when possible; unstratified fallback with a recorded reason."""
    if target:
        try:
            return stratified_patient_kfold_assignments(
                rows, folds, seed, target_column=target
            ), f"stratified on {target}"
        except CrossValidationError as exc:
            reason = f"unstratified ({exc})"
    else:
        reason = "unstratified (no usable target column)"
    patients = [str(row["patient_id"]) for row in rows]
    return patient_kfold_assignments(patients, folds, seed), reason


def _write_manifest(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    return destination.resolve()


def build_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int,
    inner_folds: int,
    seed: int,
    target: str | None,
    destination: Path,
) -> list[dict[str, Any]]:
    """Materialise one rewritten manifest per outer fold."""
    outer, outer_reason = _assign(rows, folds, seed, target)
    plan: list[dict[str, Any]] = []
    for fold in range(folds):
        test_patients = {p for p, assigned in outer.items() if assigned == fold}
        remaining = [row for row in rows if str(row["patient_id"]) not in test_patients]
        inner, inner_reason = _assign(remaining, inner_folds, seed * 100 + fold, target)
        validation_patients = {p for p, assigned in inner.items() if assigned == 0}
        rewritten: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in rows:
            patient = str(row["patient_id"])
            split = (
                "test" if patient in test_patients
                else "validation" if patient in validation_patients
                else "train"
            )
            counts[split] += 1
            rewritten.append({**row, "split": split})
        path = _write_manifest(rewritten, destination / f"fold_{fold}" / "manifest.csv")
        plan.append({
            "fold": fold,
            "manifest": str(path),
            "rows": dict(counts),
            "patients": {
                "test": len(test_patients),
                "validation": len(validation_patients),
                "train": len({str(row["patient_id"]) for row in rows})
                         - len(test_patients) - len(validation_patients),
            },
            "outer_assignment": outer_reason,
            "inner_assignment": inner_reason,
        })
    return plan


def _run(command: Sequence[str], logger: RunLogger) -> int:
    logger.log("exec: " + " ".join(command))
    return subprocess.call(list(command), cwd=str(ROOT))


def _fold_metrics(run_dir: Path) -> dict[str, float]:
    result = run_dir / "result.json"
    if not result.is_file():
        return {}
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    metrics = ((payload.get("evaluation") or {}).get("metrics") or {})
    return {
        name: float(values["value"])
        for name, values in metrics.items()
        if isinstance(values, Mapping) and values.get("value") is not None
    }


def _aggregate(per_fold: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Mean and sample standard deviation across folds, per metric."""
    names = sorted({name for fold in per_fold for name in fold})
    summary: dict[str, Any] = {}
    for name in names:
        values = [fold[name] for fold in per_fold if name in fold]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = (
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1 else 0.0
        )
        summary[name] = {
            "mean": mean,
            "std": variance ** 0.5,
            "folds": len(values),
            "values": values,
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/tasks/train_cv.py",
        description="Patient-level stratified K-fold CV around the unchanged task trainer",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--folds", type=int, default=None, help="override evaluation.cross_validation.folds")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="write fold manifests and print the plan; train nothing")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    config = load_config(args.config, args.overrides)
    paths = ProjectPaths.resolve(config)

    evaluation = dict(config.get("evaluation") or {})
    cv_config = dict(evaluation.get("cross_validation") or {})
    strategy = str(evaluation.get("split_strategy", "patient_holdout"))
    folds = int(args.folds or cv_config.get("folds", 0))
    if not cv_config.get("enabled") and args.folds is None:
        raise SystemExit(
            "cross-validation is not enabled for this config.\n"
            "  set evaluation.cross_validation.enabled=true (and split_strategy=patient_stratified_cv),\n"
            "  or pass --folds N to override explicitly"
        )
    if folds < 2:
        raise SystemExit(f"cross-validation needs at least 2 folds; got {folds}")
    inner_folds = int(cv_config.get("nested_inner_folds", 3))
    seed = int(config.get("seed", 42))

    experiment = dict(config["experiment"])
    base_id = str(experiment["id"])
    family = str(experiment.get("family") or experiment["stage"])
    output_root = paths.assert_persistent_output()
    cv_root = (output_root / "cross_validation" / family / base_id).resolve()
    cv_root.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(cv_root / "logs" / "run.log")
    logger.log(f"cv experiment={base_id} folds={folds} inner_folds={inner_folds} seed={seed}")
    logger.log(f"command={' '.join(sys.argv)}")
    logger.log(f"split_strategy={strategy} cv_type={cv_config.get('type')}")

    manifest = resolve_manifest(config, paths)
    rows = read_rows(manifest)
    if not rows:
        raise SystemExit(f"manifest is empty: {manifest}")
    columns = {key for row in rows for key in row}
    if "patient_id" not in columns:
        raise SystemExit(f"manifest has no patient_id column: {manifest}")
    target = _stratification_target(config, columns)
    logger.log(f"manifest={manifest} rows={len(rows)} stratify_on={target}")

    plan = build_folds(
        rows, folds=folds, inner_folds=inner_folds, seed=seed,
        target=target, destination=cv_root / "folds",
    )
    for entry in plan:
        logger.log(
            f"fold={entry['fold']} rows={entry['rows']} patients={entry['patients']} "
            f"outer={entry['outer_assignment']} inner={entry['inner_assignment']}"
        )
    (cv_root / "fold_plan.json").write_text(
        json.dumps({"folds": plan, "stratify_on": target, "seed": seed}, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        logger.log("cv status=dry_run_complete")
        print(f"fold manifests written to {cv_root / 'folds'}")
        return 0

    per_fold: list[dict[str, float]] = []
    failures: list[dict[str, Any]] = []
    for entry in plan:
        fold = entry["fold"]
        fold_id = f"{base_id}__cvfold{fold}"
        overrides = [
            *args.overrides,
            f"data.manifest={entry['manifest']}",
            f"experiment.id={fold_id}",
            f"data.fold={fold}",
            "experiment.variant_stamp=false",
        ]
        common = ["--config", str(args.config)]
        for override in overrides:
            common += ["--set", override]
        if args.gpus:
            common += ["--gpus", args.gpus]

        train = [sys.executable, str(ROOT / "tools" / "tasks" / "train_task.py"), *common]
        if args.overwrite:
            train.append("--overwrite")
        code = _run(train, logger)
        if code != 0:
            logger.log(f"fold={fold} status=train_failed exit={code}")
            failures.append({"fold": fold, "stage": "train", "exit_code": code})
            continue

        evaluate = [
            sys.executable, str(ROOT / "tools" / "tasks" / "evaluate.py"), *common, "--allow-full"
        ]
        code = _run(evaluate, logger)
        if code != 0:
            logger.log(f"fold={fold} status=evaluate_failed exit={code}")
            failures.append({"fold": fold, "stage": "evaluate", "exit_code": code})
            continue

        run_dir = (output_root / _family_path(family) / fold_id).resolve()
        metrics = _fold_metrics(run_dir)
        entry["run_dir"] = str(run_dir)
        entry["metrics"] = metrics
        per_fold.append(metrics)
        logger.log(f"fold={fold} status=done metrics={sorted(metrics)}")

    summary = {
        "experiment": base_id,
        "folds": folds,
        "inner_folds": inner_folds,
        "seed": seed,
        "stratify_on": target,
        "split_strategy": strategy,
        "cross_validation_type": cv_config.get("type"),
        "completed_folds": len(per_fold),
        "failures": failures,
        "per_fold": plan,
        "aggregate": _aggregate(per_fold),
        "elapsed_min": (time.perf_counter() - started) / 60,
    }
    (cv_root / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    _write_summary_markdown(summary, cv_root / "CV_SUMMARY.md")
    logger.log(f"cv status=finished completed={len(per_fold)}/{folds} failures={len(failures)}")
    print(f"CV summary written to {cv_root}")
    return 0 if not failures else 1


def _family_path(family: str) -> str:
    from source.engine.experiment import FAMILY_PATHS

    return FAMILY_PATHS.get(family, family)


def _write_summary_markdown(summary: Mapping[str, Any], destination: Path) -> None:
    lines = [
        f"# Cross-validation — `{summary['experiment']}`",
        "",
        f"{summary['folds']}-fold patient-level CV · stratified on `{summary['stratify_on']}` · "
        f"seed {summary['seed']} · {summary['completed_folds']}/{summary['folds']} fold hoàn tất",
        "",
        "| Metric | Mean | Std | Per-fold |",
        "|---|---:|---:|---|",
    ]
    for name, values in sorted((summary.get("aggregate") or {}).items()):
        per_fold = ", ".join(f"{value:.4f}" for value in values["values"])
        lines.append(f"| `{name}` | {values['mean']:.4f} | {values['std']:.4f} | {per_fold} |")
    lines += ["", "| Fold | Train | Validation | Test | Manifest |", "|---:|---:|---:|---:|---|"]
    for entry in summary.get("per_fold", []):
        rows = entry.get("rows", {})
        lines.append(
            f"| {entry['fold']} | {rows.get('train', 0)} | {rows.get('validation', 0)} | "
            f"{rows.get('test', 0)} | `{entry['manifest']}` |"
        )
    if summary.get("failures"):
        lines += ["", "## Fold lỗi", ""]
        for failure in summary["failures"]:
            lines.append(f"- fold {failure['fold']} — {failure['stage']} exit {failure['exit_code']}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
