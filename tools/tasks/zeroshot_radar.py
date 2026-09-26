"""Run the released RADAR over a cohort unchanged -- zero-shot, no training, no fine-tuning.

RADAR (alibaba-damo-academy/damo-radar) is a vision-language generalist for abdominal CT.
It ships no PE finding, so this tool scores RADAR's own pulmonary-artery organ token against
the [negative, positive] text pair in ``radar.prompts`` -- an out-of-scope probe, reported
as such. Like zeroshot_penet.py it reads the raw release NIfTI (RADAR has its own input
contract) and writes the same tables as a trained diagnosis run.

RADAR pins transformers==4.25, so the model runs in its own conda env: this file (project
env) selects the studies and evaluates, and tools/tasks/zeroshot_radar_worker.py (RADAR env,
``radar.python``) does the inference. Scores are appended per study, so ``--resume``
continues an interrupted run instead of starting over.

    python tools/tasks/zeroshot_radar.py \
      --config configs/runs/01_foundation/radar_zero_shot.yaml --allow-full

Output (same tables as a trained diagnosis run, so the two can be compared row by row):
    result.csv            one row per split (validation, test), same columns as CT-FM runs
    predictions.csv       validation + test studies: y_true, y_prob, y_pred, windows,
                          organ_voxels, fallback_crop
    preview/*.png         first five validation studies: RADAR input + its organ mask
                          (TP/TN/FP/FN in the name)
    logs.txt              terminal log, including the worker's progress lines
    result.json           full metric payload, the RADAR input contract and the prompts
    resolved_config.yaml
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.manifests import read_rows
from source.data.paths import ProjectPaths
from source.engine.experiment import OutputManager, atomic_write_json
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

WORKER = Path(__file__).with_name("zeroshot_radar_worker.py")
SCORES_FILE = "radar_scores.partial.jsonl"     # removed once predictions.csv is written
JOB_FILE = ".radar_job.json"
RADAR_CONTRACT = {
    "source": "RADAR_inference/inference_demo.py (released inference path), one organ-level finding",
    "orientation": "LAS as stored (INSPECT and the demo case); other orientations reoriented to LAS",
    "resample_mm": [1.0, 1.0, 5.0],
    "clip_hu": [-300.0, 400.0],
    "normalize": "per-volume min-max after clipping, then crop the non-zero box (+5 d, +20 h/w voxels)",
    "pad_to": [96, 256, 384],
    "window": {"roi": [96, 256, 384], "overlap": 0.25},
    "organ_feature": "RADAR's own segmentation head locates the organ; its query token attends to "
                     "the organ's image tokens in the first window where the organ is intact, else "
                     "in a crop centred on the organ (demo fallback)",
    "score": "softmax([negative, positive] similarity / temperature)[positive]",
    "text_pair": "negative = mean of radar.prompts.negative, positive = mean of radar.prompts.positive "
                 "(unit-normalised sentence embeddings, mean left unnormalised as in "
                 "infer_text_embedding_radar.pt)",
}


def _restriction(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    rows = read_rows(path)
    if not rows or "patient_id" not in rows[0]:
        raise SystemExit(f"restriction file needs rows and a patient_id column: {path}")
    return {str(row["patient_id"]) for row in rows}


def _raw_series_path(paths: ProjectPaths, config: dict[str, Any], study_id: str) -> Path:
    """The read-only NIfTI for a study; the manifest's image_path is the derived cache."""
    root = dict(config.get("radar") or {}).get("release_root")
    if root:
        release = Path(str(root))
    else:
        if paths.raw_inspect_root is None:
            raise SystemExit("raw INSPECT root is not configured; set PE_RAW_INSPECT_ROOT")
        release = paths.raw_inspect_root / "CT" / "full"
    return release / "CTPA" / f"{study_id}.nii.gz"


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


def _radar_python(configured: Any) -> Path:
    if configured:
        candidate = Path(str(configured))
    elif os.environ.get("RADAR_PYTHON"):
        candidate = Path(os.environ["RADAR_PYTHON"])
    else:
        base = shutil.which("conda")
        conda_root = Path(base).resolve().parents[1] if base else Path.home() / "miniconda3"
        candidate = conda_root / "envs" / "radar" / "bin" / "python"
    if not candidate.is_file():
        raise SystemExit(f"RADAR env python not found: {candidate}; build the env from the RADAR "
                         "requirements.txt and set radar.python or RADAR_PYTHON")
    return candidate


def _run_worker(python: Path, job: dict[str, Any], run_dir: Path, gpu: int | None, logger: RunLogger) -> None:
    job_path = run_dir / JOB_FILE
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    environment = {**os.environ, "PYTHONWARNINGS": "ignore", "TOKENIZERS_PARALLELISM": "false"}
    environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    process = subprocess.Popen(
        [str(python), str(WORKER), "--job", str(job_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if line.strip():
                logger.log(f"radar | {line.rstrip()}")
        process.wait()
    except BaseException:
        # Ctrl+C here must not leave an orphan worker holding the GPU; scores written so far stay.
        process.terminate()
        process.wait(timeout=60)
        raise
    if process.returncode:
        raise SystemExit(f"RADAR worker failed (exit {process.returncode}); see {run_dir / 'logs.txt'}")
    job_path.unlink(missing_ok=True)


def _read_scores(path: Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                scores[str(record["study_id"])] = record      # a later line wins (resume retry)
    return scores


def _outcome(y_true: int, y_pred: int) -> str:
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(int(y_true), int(y_pred))]


def main() -> int:
    parser = base_parser("Zero-shot RADAR inference over a built cohort")
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: radar.checkpoint")
    parser.add_argument("--radar-python", type=Path, default=None, help="default: radar.python")
    parser.add_argument("--label-column", default=None, help="default: task.primary_target")
    parser.add_argument("--restrict-to", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--preview-patients", type=int, default=5,
                        help="first N validation studies to render with RADAR's organ mask (default: 5)")
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
    # Collision-checked like every other run; --resume reuses the per-study scores of an
    # interrupted run, --overwrite (OVERWRITE=1 in the wrapper) starts clean.
    run_dir = manager.prepare(family, str(experiment["id"]), resume=args.resume,
                              overwrite=args.overwrite, directories=())
    manager.write_config(run_dir, {**config, "resolved_paths": paths.as_dict()}, compatibility_copy=False)
    logger = RunLogger(run_dir / "logs.txt")
    logger.log(BAR)
    logger.log(f"ZERO-SHOT {experiment['id']} | RADAR released weights, no training")
    logger.log(BAR)

    radar_config = dict(config.get("radar") or {})
    repo = paths.code_root / str(radar_config.get("repo") or "third_party/repos/damo-radar")
    checkpoint = args.checkpoint or Path(str(radar_config.get("checkpoint") or ""))
    if not str(checkpoint):
        raise SystemExit("no RADAR checkpoint: pass --checkpoint or set radar.checkpoint")
    if not checkpoint.is_absolute():
        checkpoint = paths.code_root / checkpoint
    if not checkpoint.is_file() or not (repo / "RADAR_inference" / "inference_demo.py").is_file():
        raise SystemExit(f"RADAR repository or checkpoint missing: {repo} | {checkpoint}")
    python = _radar_python(args.radar_python or radar_config.get("python"))
    organ, finding = str(radar_config["organ"]), str(radar_config["finding"])
    prompts = {side: [str(text) for text in (radar_config.get("prompts") or {}).get(side) or []]
               for side in ("negative", "positive")}
    if not prompts["negative"] or not prompts["positive"]:
        raise SystemExit("radar.prompts needs at least one negative and one positive sentence")
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    label_column = args.label_column or str((config.get("task") or {}).get("primary_target") or "")
    if not label_column:
        raise SystemExit("no label column: pass --label-column or set task.primary_target")
    manifest_rows = read_rows(resolve_manifest(config, paths))
    restrict = _restriction(args.restrict_to)
    evaluation = dict(config.get("evaluation") or {})
    devices = [int(value) for value in (config.get("compute") or {}).get("devices") or []]
    accelerator = str((config.get("compute") or {}).get("accelerator") or "cuda")
    gpu = devices[0] if devices and accelerator != "cpu" else None
    device = "cpu" if gpu is None else "cuda:0"          # the worker sees only CUDA_VISIBLE_DEVICES=gpu
    logger.log(f"radar repo={repo} commit={commit} checkpoint={checkpoint} python={python} "
               f"gpu={gpu if gpu is not None else 'none (CPU, slow)'} item={organ}_{finding}")

    selected: dict[str, list[dict[str, Any]]] = {}
    for split in ("validation", "test"):
        rows = _rows_for_split(manifest_rows, split, label_column, restrict)
        selected[split] = rows[: int(args.max_cases)] if args.max_cases is not None else rows
    scores_path = run_dir / SCORES_FILE
    scores = _read_scores(scores_path)
    todo = [row for split in ("validation", "test") for row in selected[split]
            if row["study_id"] not in scores or "error" in scores[row["study_id"]]]
    logger.log(f"studies validation={len(selected['validation'])} test={len(selected['test'])} "
               f"already_scored={len(scores) - sum('error' in item for item in scores.values())} to_score={len(todo)}")
    base_job = {"repo": str(repo), "checkpoint": str(checkpoint), "device": device,
                "item": f"{organ}_{finding}", "organ": organ, "organ_label": radar_config.get("organ_label", organ),
                "prompts": prompts}
    if todo:
        _run_worker(python, {**base_job, "mode": "score", "scores_path": str(scores_path),
                             "studies": [{"study_id": row["study_id"],
                                          "path": str(_raw_series_path(paths, config, row["study_id"]))}
                                         for row in todo]}, run_dir, gpu, logger)
        scores = _read_scores(scores_path)

    results: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    for split in ("validation", "test"):
        for row in selected[split]:
            record = scores.get(row["study_id"], {"error": "no score written"})
            if "y_prob" not in record:
                skipped.append({"study_id": row["study_id"], "split": split, "error": str(record.get("error"))})
                continue
            row.update({"y_prob": float(record["y_prob"]), "windows": record.get("windows"),
                        "organ_voxels": record.get("organ_voxels"), "fallback_crop": record.get("fallback_crop")})
        results[split] = [row for row in selected[split] if "y_prob" in row]
        logger.log(f"split={split} scored={len(results[split])} "
                   f"skipped={sum(item['split'] == split for item in skipped)}")
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

    preview_rows = []
    for index, row in enumerate(results["validation"][: args.preview_patients], start=1):
        y_pred = int(row["y_prob"] >= threshold)
        outcome = _outcome(row["y_true"], y_pred)
        preview_rows.append({
            "study_id": row["study_id"], "patient_id": row["patient_id"], "y_true": row["y_true"],
            "y_pred": y_pred, "outcome": outcome,
            "path": str(_raw_series_path(paths, config, row["study_id"])),
            "png": str(run_dir / "preview" / f"{index:02d}_{row['patient_id']}_{row['study_id']}_{outcome}.png"),
        })
    preview_report: dict[str, Any] = {"status": "skipped", "patients": 0, "files": [], "errors": []}
    if preview_rows:
        try:
            _run_worker(python, {**base_job, "mode": "preview", "threshold": threshold,
                                 "threshold_rule": threshold_rule(threshold_source, threshold_method),
                                 "studies": preview_rows}, run_dir, gpu, logger)
            files = [Path(row["png"]).name for row in preview_rows if Path(row["png"]).is_file()]
            preview_report = {"status": "completed" if len(files) == len(preview_rows) else "partial",
                              "patients": len(files), "files": files,
                              "errors": [row["study_id"] for row in preview_rows if Path(row["png"]).name not in files]}
        except SystemExit as exc:          # previews are a check, never a reason to lose the metrics
            preview_report = {"status": "failed", "patients": 0, "files": [], "errors": [str(exc)]}
    logger.log(f"preview status={preview_report['status']} studies={preview_report['patients']} "
               f"dir={run_dir / 'preview'} files={','.join(preview_report['files'])}")

    point = {name: item["value"] for name, item in test_metrics.items()}
    point["threshold"] = threshold
    fallback_count = sum(bool(row.get("fallback_crop")) for split in results.values() for row in split)
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
        "centre_crop_fallbacks": fallback_count,
        "label_column": label_column,
        "restrict_to": str(args.restrict_to) if args.restrict_to else None,
    }
    atomic_write_json(run_dir / "result.json", {
        "experiment": {**experiment, "status": "completed"},
        "zero_shot": {
            "model": "RADAR",
            "trained_here": False,
            "checkpoint": str(checkpoint),
            "repo": str(repo),
            "repo_commit": commit,
            "item": f"{organ}_{finding}",
            "organ": organ,
            "organ_label": radar_config.get("organ_label", organ),
            "prompts": prompts,
            "prompt_origin": "written for this project; RADAR ships no PE finding",
            "input_contract": RADAR_CONTRACT,
            "reads": "raw release NIfTI, not the shared volumes/*.npy cache",
        },
        "evaluation": result_evaluation,
        "preview": preview_report,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    scores_path.unlink(missing_ok=True)      # its content now lives in predictions.csv / result.json
    logger.log(f"written: {run_dir / 'result.csv'} | {run_dir / 'predictions.csv'} | {run_dir / 'preview'} | {run_dir / 'result.json'}")
    logger.log(final_evaluation_block(str(experiment["id"]), result_evaluation, run_dir))
    logger.log("zeroshot_radar status=finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
