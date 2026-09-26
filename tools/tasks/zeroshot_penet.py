"""Run the released PENet over a cohort unchanged -- zero-shot, no training, no fine-tuning.

PENet ships as a trained CTPA PE classifier, so this is a real external baseline for the
diagnosis task. It deliberately does not touch the shared CT cache: PENet has its own
input contract (non-overlapping 32-slice windows, 208x208, its own HU window), so the tool
reads the raw NIfTI of the release and reproduces the repository's test-time transform.
Per series it takes sigmoid of each window logit and keeps the maximum, which is what
``third_party/repos/penet/test.py`` does.

    python tools/tasks/zeroshot_penet.py \
      --config configs/runs/01_foundation/penet_zero_shot.yaml --allow-full

Output (same tables as a trained diagnosis run, so the two can be compared row by row):
    result.csv            one row per split (validation, test), same columns as CT-FM runs
    predictions.csv       validation + test series: y_true, y_prob, y_pred, windows
    preview/*.{html,png}  Grad-CAM of the first five validation studies (TP/TN/FP/FN in the name)
    logs.txt              terminal log
    result.json           full metric payload and the PENet input contract
    resolved_config.yaml
"""
from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from source.components.encoders.image.penet_zeroshot import (
    PENET_CONTRACT,
    PenetError,
    load_penet,
    read_series_hu,
    series_windows,
)
from source.data.manifests import read_rows
from source.data.paths import ProjectPaths
from source.engine.experiment import OutputManager, atomic_write_json
from source.imaging.penet_preview import write_penet_previews
from source.metrics.result_table import (
    log_lines,
    metric_bundle,
    prediction_rows,
    result_rows,
    select_threshold,
    split_summary,
    threshold_rule,
    write_predictions_csv,
    write_result_csv,
)
from source.utils.console import BAR, final_evaluation_block
from source.utils.logger import RunLogger
from tools._common import base_parser, resolve_cli_config, resolve_manifest

PROGRESS_EVERY = 25       # studies between "scored N/total" log lines


def _restriction(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    rows = read_rows(path)
    if not rows or "patient_id" not in rows[0]:
        raise SystemExit(f"restriction file needs rows and a patient_id column: {path}")
    return {str(row["patient_id"]) for row in rows}


def _raw_series_path(paths: ProjectPaths, config: dict[str, Any], study_id: str) -> Path:
    """The read-only NIfTI for a study; the manifest's image_path is the derived cache."""
    source = dict(config.get("penet") or {})
    root = source.get("release_root")
    if root:
        release = Path(str(root))
    else:
        if paths.raw_inspect_root is None:
            raise SystemExit("raw INSPECT root is not configured; set PE_RAW_INSPECT_ROOT")
        release = paths.raw_inspect_root / "CT" / "full"
    return release / "CTPA" / f"{study_id}.nii.gz"


@torch.no_grad()
def _series_probability(model, volume, device, aggregate: str) -> tuple[float, int]:
    probabilities: list[float] = []
    for window in series_windows(volume):
        logit = model(torch.from_numpy(window).unsqueeze(0).to(device))
        probabilities.append(float(torch.sigmoid(logit).flatten()[0]))
    if not probabilities:
        raise PenetError("series produced no windows")
    value = max(probabilities) if aggregate == "max" else sum(probabilities) / len(probabilities)
    return value, len(probabilities)


def _rows_for_split(manifest_rows, split, label_column, restrict):
    selected = []
    for row in manifest_rows:
        if str(row.get("split")) != split:
            continue
        if restrict is not None and str(row["patient_id"]) not in restrict:
            continue
        raw = str(row.get(label_column, "")).strip().upper()
        if raw in {"TRUE", "1"}:
            label = 1
        elif raw in {"FALSE", "0"}:
            label = 0
        else:
            continue          # CENSORED / MISSING is not a binary target
        selected.append({"patient_id": str(row["patient_id"]), "study_id": str(row["study_id"]),
                         "y_true": label})
    return selected


def main() -> int:
    parser = base_parser("Zero-shot PENet inference over a built cohort")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=Path("third_party/repos/penet"))
    parser.add_argument("--label-column", default=None, help="default: task.primary_target")
    parser.add_argument("--restrict-to", type=Path, default=None)
    parser.add_argument("--aggregate", choices=("max", "mean"), default="max")
    parser.add_argument("--slice-order", choices=("superior_to_inferior", "inferior_to_superior"),
                        default="superior_to_inferior")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--preview-patients", type=int, default=5,
                        help="first N validation studies to render as Grad-CAM previews (default: 5)")
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    if not args.allow_full and args.max_cases is None:
        raise SystemExit("zero-shot inference needs an explicit scope: --allow-full or --max-cases N")
    if args.preview_patients < 0:
        raise SystemExit("--preview-patients must be nonnegative")

    config = resolve_cli_config(args)
    paths = ProjectPaths.resolve(config)
    manager = OutputManager(paths)
    experiment = dict(config["experiment"])
    family = str(experiment.get("family") or experiment["stage"])
    # Collision-checked like every other run: an existing result is never silently
    # overwritten; pass --overwrite (OVERWRITE=1 in the wrapper) to replace it.
    run_dir = manager.prepare(family, str(experiment["id"]), overwrite=args.overwrite, directories=())
    manager.write_config(run_dir, {**config, "resolved_paths": paths.as_dict()}, compatibility_copy=False)
    logger = RunLogger(run_dir / "logs.txt")
    logger.log(BAR)
    logger.log(f"ZERO-SHOT {experiment['id']} | PENet released weights, no training")
    logger.log(BAR)

    penet_config = dict(config.get("penet") or {})
    checkpoint = args.checkpoint or Path(str(penet_config.get("checkpoint") or ""))
    if not str(checkpoint):
        raise SystemExit("no PENet checkpoint: pass --checkpoint or set penet.checkpoint")
    if not checkpoint.is_absolute():
        checkpoint = paths.code_root / checkpoint
    repo = args.repo if args.repo.is_absolute() else paths.code_root / args.repo

    label_column = args.label_column or str((config.get("task") or {}).get("primary_target") or "")
    if not label_column:
        raise SystemExit("no label column: pass --label-column or set task.primary_target")

    manifest = resolve_manifest(config, paths)
    manifest_rows = read_rows(manifest)
    restrict = _restriction(args.restrict_to)
    evaluation = dict(config.get("evaluation") or {})

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus else "cpu")
    model, model_meta = load_penet(checkpoint, repo)
    model.to(device)
    logger.log(f"penet checkpoint={checkpoint} device={device} meta={model_meta}")

    selected: dict[str, list[dict[str, Any]]] = {}
    for split in ("validation", "test"):
        rows = _rows_for_split(manifest_rows, split, label_column, restrict)
        selected[split] = rows[: int(args.max_cases)] if args.max_cases is not None else rows
    total = sum(len(rows) for rows in selected.values())
    logger.log(f"studies validation={len(selected['validation'])} test={len(selected['test'])} total={total}")

    results: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    done, began = 0, time.perf_counter()
    for split in ("validation", "test"):
        rows = selected[split]
        for row in rows:
            path = _raw_series_path(paths, config, row["study_id"])
            try:
                volume = read_series_hu(path, args.slice_order)
                probability, windows = _series_probability(model, volume, device, args.aggregate)
                row["y_prob"], row["windows"] = probability, windows
            except (PenetError, RuntimeError, OSError) as exc:
                skipped.append({"study_id": row["study_id"], "error": f"{type(exc).__name__}: {exc}"})
                logger.log(f"study {row['study_id']} skipped: {skipped[-1]['error']}")
            done += 1
            if done % PROGRESS_EVERY == 0 or done == total:
                rate = (time.perf_counter() - began) / done
                logger.log(f"scored {done}/{total} split={split} ({rate:.1f} s/study, "
                           f"~{rate * (total - done) / 3600:.1f} h left)")
        results[split] = [row for row in rows if "y_prob" in row]
        logger.log(f"split={split} scored={len(results[split])} skipped={len(skipped)}")
        if not results[split]:
            raise SystemExit(f"no scorable series in split={split}")

    threshold_method = str(evaluation.get("threshold_method", "youden"))
    threshold, threshold_source = select_threshold(results["validation"], threshold_method)
    logger.log(
        f"threshold={threshold:.6f} rule={threshold_rule(threshold_source, threshold_method)} "
        "(chosen on validation, applied unchanged to test)"
    )
    samples = int(evaluation.get("bootstrap_samples", 2000))
    confidence = float(evaluation.get("confidence", 0.95))
    seed = int(config.get("seed", 42))
    validation_metrics, _ = metric_bundle(results["validation"], "diagnosis", threshold, with_ci=False)
    test_metrics, ci_reason = metric_bundle(
        results["test"], "diagnosis", threshold,
        with_ci=True, samples=samples, confidence=confidence, seed=seed,
    )
    split_entries: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    split_metrics: dict[str, Any] = {"validation": validation_metrics, "test": test_metrics}
    for split in ("validation", "test"):
        summary = split_summary(results[split], threshold)
        split_metrics[f"{split}_rows"] = summary["n_studies"]
        split_metrics[f"{split}_patients"] = summary["n_patients"]
        split_metrics[f"{split}_positives"] = summary["n_pos"]
        split_metrics[f"{split}_negatives"] = summary["n_neg"]
        for line in log_lines(label_column, split, summary, threshold=threshold):
            logger.log(line)
        split_entries[split] = {
            "metrics": split_metrics[split],
            "summary": summary,
            "ci_reason": ci_reason if split == "test" else None,
        }
        predictions.extend(prediction_rows(split, label_column, results[split], threshold))
    write_result_csv(
        run_dir / "result.csv",
        result_rows(
            experiment=str(experiment["id"]),
            target=label_column,
            splits=split_entries,
            threshold=threshold,
            threshold_source=threshold_source,
            stage="diagnosis",
            threshold_method=threshold_method,
        ),
        stage="diagnosis",
    )
    write_predictions_csv(run_dir / "predictions.csv", predictions)
    preview_report = write_penet_previews(
        [{**row, "y_pred": int(float(row["y_prob"]) >= threshold)}
         for row in results["validation"]],
        run_dir / "preview",
        lambda study_id: _raw_series_path(paths, config, study_id),
        model=model,
        device=device,
        threshold=threshold,
        threshold_rule=threshold_rule(threshold_source, threshold_method),
        maximum_patients=args.preview_patients,
        slice_order=args.slice_order,
        aggregate=args.aggregate,
        checkpoint=checkpoint,
        target=label_column,
    )
    logger.log(
        f"preview status={preview_report['status']} studies={preview_report['patients']} "
        f"dir={run_dir / 'preview'} cam={','.join(preview_report['cam_status'])} "
        f"max_abs_dp={preview_report['max_abs_probability_delta']} files={','.join(preview_report['files'])}"
    )
    for error in preview_report["errors"]:
        logger.log(f"preview error {error}")
    point = {name: item["value"] for name, item in test_metrics.items()}
    point["threshold"] = threshold
    result_evaluation = {
        "primary_target": label_column,
        "threshold": threshold,
        "threshold_split": "validation",
        "threshold_method": threshold_method,
        "threshold_source": threshold_source,
        "threshold_rule": threshold_rule(threshold_source, threshold_method),
        "bootstrap": {"unit": "patient", "samples": samples, "confidence": confidence},
        "point_metrics": point,
        "metrics": test_metrics,
        "split_metrics": {label_column: split_metrics},
        "bootstrap_unavailable_reason": ci_reason,
        "evaluated_patients": split_metrics["test_patients"],
        "scored": {split: len(items) for split, items in results.items()},
        "skipped_series": skipped[:200],
        "skipped_count": len(skipped),
        "label_column": label_column,
        "restrict_to": str(args.restrict_to) if args.restrict_to else None,
    }
    atomic_write_json(run_dir / "result.json", {
        "experiment": {**experiment, "status": "completed"},
        "zero_shot": {
            "model": "PENet",
            "trained_here": False,
            "checkpoint": str(checkpoint),
            "repo": str(repo),
            **model_meta,
            "input_contract": PENET_CONTRACT,
            "slice_order": args.slice_order,
            "aggregate": args.aggregate,
            "reads": "raw release NIfTI, not the shared volumes/*.npy cache",
        },
        "evaluation": result_evaluation,
        "preview": preview_report,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    logger.log(f"written: {run_dir / 'result.csv'} | {run_dir / 'predictions.csv'} | {run_dir / 'preview'} | {run_dir / 'result.json'}")
    logger.log(final_evaluation_block(str(experiment["id"]), result_evaluation, run_dir))
    logger.log("zeroshot_penet status=finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
