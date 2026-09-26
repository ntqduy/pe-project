"""Rebuild the Grad-CAM preview of an evaluated diagnosis/prognosis run, without re-evaluating.

evaluate.py already writes the preview at the end of every evaluation. This tool redoes only
that step for an existing run: it loads the run's resolved config and checkpoint, the
validation threshold recorded by evaluate in result.json (never re-selected here) and the
validation probabilities in predictions.csv, then writes the same files as evaluate.

    source scripts/use_gcs_storage.sh
    python tools/tasks/gradcam_preview.py \
      --run-dir /mnt/pe-storage/pe-project/outputs/diagnosis/DX_ctfm_frozen__ds_smoke_30/epoch_30

--run-dir names one epoch_<E>/ bundle; given the run folder above it, the highest E is used.
Output (default: <bundle>/preview/, replacing the previous preview files):
    NN_<patient>_<study>_<TP|TN|FP|FN>.html   offline viewer: every input slice, CT | CT + CAM
    NN_<patient>_<study>_<TP|TN|FP|FN>.png    summary: header + montage of the 8 top-CAM slices
With the default output the preview lines are appended to <bundle>/logs.txt as well.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from source.data.paths import ProjectPaths
from source.engine.checkpoint import checkpoint_sha256, load_checkpoint
from source.engine.factory import build_task_model
from source.engine.task_artifacts import (
    CAM_METHODS,
    latest_epoch_directory,
    preview_log_lines,
    write_backbone_previews,
)
from source.utils.config import load_config
from source.utils.logger import RunLogger
from tools._common import build_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="outputs/<family>/<RUN_ID>[/epoch_<E>]"
    )
    parser.add_argument("--checkpoint", type=Path, help="default: <bundle>/checkpoint/best.ckpt")
    parser.add_argument("--output", type=Path, help="default: <bundle>/preview")
    parser.add_argument("--maximum-patients", type=int, default=5)
    parser.add_argument("--method", choices=sorted(CAM_METHODS), default="hirescam")
    parser.add_argument("--device", default=None, help="torch device (default: cuda when available)")
    args = parser.parse_args()
    if args.maximum_patients < 0:
        parser.error("--maximum-patients must be nonnegative")

    run_dir = args.run_dir.resolve()
    epoch_dir = run_dir if (run_dir / "checkpoint").is_dir() else latest_epoch_directory(run_dir)
    if epoch_dir is None:
        raise SystemExit(f"no epoch_<E>/ bundle in {run_dir}")
    # A bundle holds its own config and result.json; older run folders kept them one level up.
    if (epoch_dir / "resolved_config.yaml").is_file():
        run_dir = epoch_dir
    checkpoint = (args.checkpoint or epoch_dir / "checkpoint" / "best.ckpt").resolve()
    output = (args.output or epoch_dir / "preview").resolve()
    config = load_config(run_dir / "resolved_config.yaml", [])
    stage = str(config["experiment"]["stage"])
    if stage in {"ablation", "roi_student"}:
        stage = str((config.get("task") or {}).get("base_stage") or stage)
    if stage not in {"diagnosis", "prognosis"}:
        raise SystemExit(f"preview needs a diagnosis/prognosis run, got stage={stage}")
    primary = str((config.get("task") or {}).get("primary_target") or "pe_present")

    # The cutoff evaluate selected on validation and applied to every split.
    evaluation = json.loads((run_dir / "result.json").read_text(encoding="utf-8")).get("evaluation") or {}
    target_result = (evaluation.get("targets") or {}).get(primary) or {}
    threshold = target_result.get("threshold", evaluation.get("threshold"))
    rule = target_result.get("threshold_rule", evaluation.get("threshold_rule"))
    if threshold is None:
        print("result.json has no evaluate threshold: predicted class is N/A", file=sys.stderr)
    reference: dict[tuple[str, str], float] = {}
    predictions = epoch_dir / "predictions.csv"
    if predictions.is_file():
        with predictions.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == "validation" and row.get("target", primary) == primary:
                    reference[(str(row["patient_id"]), str(row["study_id"]))] = float(row["y_prob"])

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = build_task_model(config)
    load_checkpoint(checkpoint, model=model, strict=True)
    model.to(device)
    dataset = build_dataset(config, ProjectPaths.resolve(config), "validation")
    report = write_backbone_previews(
        model,
        dataset,
        config,
        device,
        output,
        maximum_patients=args.maximum_patients,
        threshold=None if threshold is None else float(threshold),
        threshold_rule=None if rule is None else str(rule),
        reference_probabilities=reference,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256(checkpoint),
        cam_method=args.method,
    )
    lines = preview_log_lines(report, output)
    if args.output is None:
        logger = RunLogger(epoch_dir / "logs.txt")
        for line in lines:
            logger.log(line)
    else:
        print("\n".join(lines))
    return 0 if not report.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
