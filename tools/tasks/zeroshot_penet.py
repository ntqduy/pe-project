"""Run the released PENet over a cohort unchanged -- zero-shot, no training, no fine-tuning.

PENet ships as a trained CTPA PE classifier, so this is a real external baseline for the
diagnosis task. It deliberately does not touch the shared CT cache: PENet has its own
input contract (non-overlapping 32-slice windows, 208x208, its own HU window), so the tool
reads the raw NIfTI of the release and reproduces the repository's test-time transform.
Per series it takes sigmoid of each window logit and keeps the maximum, which is what
``third_party/repos/penet/test.py`` does.

    python tools/tasks/zeroshot_penet.py \
      --config configs/runs/01_foundation/penet_zero_shot.yaml --allow-full
"""
from __future__ import annotations

import sys
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
from source.metrics.bootstrap import bootstrap_binary_predictions
from source.metrics.classification import binary_classification_metrics, select_threshold_on_validation
from source.utils.logger import RunLogger
from tools._common import base_parser, resolve_cli_config, resolve_manifest, write_parquet_atomic


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
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    if not args.allow_full and args.max_cases is None:
        raise SystemExit("zero-shot inference needs an explicit scope: --allow-full or --max-cases N")

    config = resolve_cli_config(args)
    paths = ProjectPaths.resolve(config)
    manager = OutputManager(paths)
    experiment = dict(config["experiment"])
    run_dir = manager.run_dir(str(experiment.get("family") or experiment["stage"]), str(experiment["id"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir / "logs" / "run.log")

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

    results: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    for split in ("validation", "test"):
        rows = _rows_for_split(manifest_rows, split, label_column, restrict)
        if args.max_cases is not None:
            rows = rows[: int(args.max_cases)]
        for row in rows:
            path = _raw_series_path(paths, config, row["study_id"])
            try:
                volume = read_series_hu(path, args.slice_order)
                probability, windows = _series_probability(model, volume, device, args.aggregate)
            except (PenetError, RuntimeError, OSError) as exc:
                skipped.append({"study_id": row["study_id"], "error": f"{type(exc).__name__}: {exc}"})
                continue
            row["y_prob"], row["windows"] = probability, windows
        results[split] = [row for row in rows if "y_prob" in row]
        logger.log(f"split={split} scored={len(results[split])} skipped={len(skipped)}")
        if not results[split]:
            raise SystemExit(f"no scorable series in split={split}")

    threshold = select_threshold_on_validation(
        [r["y_true"] for r in results["validation"]],
        [r["y_prob"] for r in results["validation"]],
        str(evaluation.get("threshold_method", "youden")),
    )
    rows = results["test"]
    samples = int(evaluation.get("bootstrap_samples", 2000))
    confidence = float(evaluation.get("confidence", 0.95))
    point = binary_classification_metrics(
        [r["y_true"] for r in rows], [r["y_prob"] for r in rows], threshold
    )
    interval = bootstrap_binary_predictions(
        rows, threshold, n_bootstrap=samples, confidence=confidence, seed=int(config.get("seed", 42))
    )

    write_parquet_atomic(rows, run_dir / "predictions.parquet")
    atomic_write_json(run_dir / "result.json", {
        "experiment": experiment,
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
        "evaluation": {
            "threshold": threshold,
            "threshold_split": "validation",
            "threshold_method": str(evaluation.get("threshold_method", "youden")),
            "bootstrap": {"unit": "patient", "samples": samples, "confidence": confidence},
            "point_metrics": point,
            "metrics": interval,
            "scored": {split: len(items) for split, items in results.items()},
            "skipped_series": skipped[:200],
            "skipped_count": len(skipped),
            "label_column": label_column,
            "restrict_to": str(args.restrict_to) if args.restrict_to else None,
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    logger.log(f"zeroshot_penet status=finished auroc={point.get('auroc')}")
    print(f"zero-shot PENet written to {run_dir}")
    print(f"  auroc={point.get('auroc'):.4f}  n={len(rows)}  skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
