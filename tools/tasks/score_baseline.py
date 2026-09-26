"""Evaluate a manifest score column as a fixed clinical baseline -- no model, no training.

A clinical severity score is a baseline, not a learned arm. Fitting an encoder and a head
on top of one scalar can reorder cases and quietly turn the reference into another model,
which is not what "sPESI baseline" means in the INSPECT paper: there the raw score is
pushed through a monotone transform and scored directly.

This tool does that, and reports with the same metrics, threshold rule, patient-level
bootstrap and output layout as ``evaluate.py``, so its row sits in the same table as a
CT-FM arm.

    python tools/tasks/score_baseline.py \
      --config configs/runs/04_prognosis/modality/spesi.yaml \
      --score-column spesi --allow-full \
      --restrict-to <derived>/cache/<profile>/clinical/spesi_evaluable.csv
"""
from __future__ import annotations

import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.manifests import read_rows
from source.data.paths import ProjectPaths
from source.engine.experiment import OutputManager, atomic_write_json
from source.metrics.bootstrap import bootstrap_prognosis_predictions
from source.metrics.calibration import calibration_curve_points
from source.metrics.classification import select_threshold_on_validation
from source.metrics.prognosis import prognosis_metrics
from source.utils.logger import RunLogger
from tools._common import base_parser, resolve_cli_config, resolve_manifest, write_parquet_atomic

TRANSFORMS = ("platt", "raw_sigmoid", "minmax")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _restriction(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    rows = read_rows(path)
    if not rows or "patient_id" not in rows[0]:
        raise SystemExit(f"restriction file needs rows and a patient_id column: {path}")
    has_study = "study_id" in rows[0]
    return {(str(r["patient_id"]), str(r["study_id"]) if has_study else "") for r in rows}


def _keep(row: dict[str, Any], restrict: set[tuple[str, str]] | None) -> bool:
    if restrict is None:
        return True
    return (str(row["patient_id"]), str(row["study_id"])) in restrict or (
        str(row["patient_id"]),
        "",
    ) in restrict


def _split_rows(
    manifest_rows: list[dict[str, Any]],
    split: str,
    score_column: str,
    label_column: str,
    restrict: set[tuple[str, str]] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Rows usable for scoring in one split, plus how many were dropped as unscorable."""
    rows: list[dict[str, Any]] = []
    dropped = 0
    for row in manifest_rows:
        if str(row.get("split")) != split or not _keep(row, restrict):
            continue
        score = _finite(row.get(score_column))
        label = _finite(row.get(label_column))
        if score is None or label is None or int(label) not in (0, 1):
            dropped += 1
            continue
        rows.append(
            {
                "patient_id": str(row["patient_id"]),
                "study_id": str(row["study_id"]),
                "score": score,
                "y_true": int(label),
            }
        )
    return rows, dropped


def _probabilities(
    train: list[dict[str, Any]], target: list[dict[str, Any]], transform: str
) -> tuple[list[float], dict[str, Any]]:
    """Map the raw score onto [0,1]. Every option here is monotone, so ranking is the score's."""
    scores = [row["score"] for row in target]
    if transform == "raw_sigmoid":
        return [1.0 / (1.0 + math.exp(-value)) for value in scores], {
            "transform": "raw_sigmoid",
            "note": "sigmoid of the raw score, as in INSPECT get_model_performance.py; "
            "discrimination is meaningful, calibration is not",
        }
    if transform == "minmax":
        low = min(row["score"] for row in train)
        high = max(row["score"] for row in train)
        span = (high - low) or 1.0
        return [min(max((value - low) / span, 0.0), 1.0) for value in scores], {
            "transform": "minmax",
            "fit_split": "validation",
            "low": low,
            "high": high,
            "note": "linear rescale; discrimination is meaningful, calibration is not",
        }
    from sklearn.linear_model import LogisticRegression

    labels = {row["y_true"] for row in train}
    if labels != {0, 1}:
        raise SystemExit("platt calibration needs both classes in the validation split")
    model = LogisticRegression()
    model.fit([[row["score"]] for row in train], [row["y_true"] for row in train])
    probabilities = [float(p[1]) for p in model.predict_proba([[v] for v in scores])]
    return probabilities, {
        "transform": "platt",
        "fit_split": "validation",
        "coefficient": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "note": "one-parameter logistic calibration fit on validation only; monotone, so "
        "ranking is still the raw score's, but Brier and calibration are interpretable",
    }


def main() -> int:
    parser = base_parser("Evaluate a manifest score column as a fixed clinical baseline")
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--label-column", default=None, help="default: task.primary_target")
    parser.add_argument("--restrict-to", type=Path, default=None)
    parser.add_argument("--transform", choices=TRANSFORMS, default="platt")
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()

    config = resolve_cli_config(args)
    if not args.allow_full:
        raise SystemExit("scoring the whole split needs an explicit --allow-full")
    paths = ProjectPaths.resolve(config)
    manager = OutputManager(paths)
    experiment = dict(config["experiment"])
    family = str(experiment.get("family") or experiment["stage"])
    run_dir = manager.run_dir(family, str(experiment["id"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir / "logs" / "run.log")

    evaluation = dict(config.get("evaluation") or {})
    task = dict(config.get("task") or {})
    label_column = args.label_column or str(task.get("primary_target") or "")
    if not label_column:
        raise SystemExit("no label column: pass --label-column or set task.primary_target")

    manifest = resolve_manifest(config, paths)
    manifest_rows = read_rows(manifest)
    if manifest_rows and args.score_column not in manifest_rows[0]:
        raise SystemExit(f"{manifest} has no column {args.score_column!r}")
    restrict = _restriction(args.restrict_to)

    logger.log(f"score_baseline column={args.score_column} label={label_column} manifest={manifest}")
    counts: dict[str, Any] = {}
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("validation", "test"):
        rows, dropped = _split_rows(manifest_rows, split, args.score_column, label_column, restrict)
        splits[split] = rows
        counts[split] = {"scored": len(rows), "dropped_unscorable": dropped}
        logger.log(f"split={split} scored={len(rows)} dropped={dropped}")
        if not rows:
            raise SystemExit(f"no scorable rows in split={split}")

    probabilities, transform_report = _probabilities(
        splits["validation"], splits["validation"], args.transform
    )
    for row, probability in zip(splits["validation"], probabilities):
        row["y_prob"] = probability
    threshold = select_threshold_on_validation(
        [row["y_true"] for row in splits["validation"]],
        [row["y_prob"] for row in splits["validation"]],
        str(evaluation.get("threshold_method", "youden")),
    )

    test_probabilities, _ = _probabilities(splits["validation"], splits["test"], args.transform)
    for row, probability in zip(splits["test"], test_probabilities):
        row["y_prob"] = probability
    rows = splits["test"]

    samples = int(evaluation.get("bootstrap_samples", 2000))
    confidence = float(evaluation.get("confidence", 0.95))
    point = prognosis_metrics(
        [row["y_true"] for row in rows], [row["y_prob"] for row in rows], threshold
    )
    interval = bootstrap_prognosis_predictions(
        rows, threshold, n_bootstrap=samples, confidence=confidence, seed=int(config.get("seed", 42))
    )
    curve = calibration_curve_points(
        [row["y_true"] for row in rows],
        [row["y_prob"] for row in rows],
        bins=int(evaluation.get("calibration_bins", 10)),
        strategy=str(evaluation.get("calibration_strategy", "quantile")),
    )

    write_parquet_atomic(rows, run_dir / "predictions.parquet")
    write_parquet_atomic(curve, run_dir / "calibration_curve.parquet")
    payload = {
        "experiment": dict(config.get("experiment") or {}),
        "baseline": {
            "kind": "fixed_score",
            "score_column": args.score_column,
            "label_column": label_column,
            "manifest": str(manifest),
            "restrict_to": str(args.restrict_to) if args.restrict_to else None,
            "probability": transform_report,
            "trained": False,
        },
        "evaluation": {
            "threshold": threshold,
            "threshold_method": str(evaluation.get("threshold_method", "youden")),
            "threshold_split": "validation",
            "bootstrap": {"unit": "patient", "samples": samples, "confidence": confidence},
            "point_metrics": point,
            "metrics": interval,
            "calibration_curve": curve,
            "cases": counts,
            "evaluated_patients": len({row["patient_id"] for row in rows}),
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(run_dir / "result.json", payload)
    logger.log(f"score_baseline status=finished threshold={threshold:.6f}")
    print(f"baseline written to {run_dir}")
    print(f"  auroc={point.get('auroc'):.4f}  auprc={point.get('auprc'):.4f}  n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
