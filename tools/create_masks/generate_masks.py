from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.manifests import read_rows
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.engine.experiment import OutputManager, compact_result, prepare_resumable_run
from source.segmentation.labels import study_pe_labels
from source.segmentation.pipeline import generate_pseudo_anatomy
from source.utils.config import load_config, parse_devices
from source.utils.environment import environment_report
from source.utils.logger import RunLogger
from tools._common import select_patient_rows, write_csv_atomic, write_parquet_atomic

QC_SUMMARY_COLUMNS = (
    "patient_id",
    "study_id",
    "anatomy",
    "status",
    "voxel_count",
    "volume_ml",
    "n_components",
    "z_first",
    "z_last",
    "duplicate_of",
    "cross_model_dice",
    "reason",
    "preview_png",
    "mask_path",
)


def qc_summary_row(row: dict) -> dict:
    """Human-readable QC row; blank cells mean "not applicable" (for example no mask)."""

    def value(key: str) -> object:
        raw = row.get(key)
        return "" if raw is None else raw

    volume = row.get("volume_ml")
    dice = row.get("cross_model_dice")
    summary = {
        "patient_id": value("patient_id"),
        "study_id": value("study_id"),
        "anatomy": value("anatomy"),
        "status": value("status"),
        "voxel_count": value("voxel_count"),
        "volume_ml": "" if volume is None else round(float(volume), 2),
        "n_components": value("component_count"),
        "z_first": value("z_first"),
        "z_last": value("z_last"),
        "duplicate_of": value("duplicate_of"),
        "cross_model_dice": "" if dice is None else round(float(dice), 4),
        "reason": value("reason"),
        "preview_png": value("preview_png"),
        "mask_path": value("mask_path"),
    }
    return {key: summary[key] for key in QC_SUMMARY_COLUMNS}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo-anatomy masks, QC, and previews; "
            "never trains a segmentation model"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--patient-id",
        dest="patient_ids",
        action="append",
        metavar="ID",
        help="process every study for this patient; repeat for more patients",
    )
    selection.add_argument("--max-cases", type=int, help="process the first N manifest studies")
    selection.add_argument("--allow-full", action="store_true", help="process the complete manifest")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, repeatable; this is how the dataset profile is selected",
    )
    parser.add_argument("--gpus", help="physical GPU IDs for independent study workers, e.g. 0,1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    config = load_config(args.config, args.overrides)
    config["overwrite"] = bool(args.overwrite)
    config["resume"] = not args.overwrite
    paths = ProjectPaths.resolve(config)
    segmentation = dict(config.get("segmentation") or {})
    if segmentation.get("repository"):
        segmentation["repository"] = str(paths.code_asset(segmentation["repository"]))
    if segmentation.get("weights_directory"):
        segmentation["weights_directory"] = str(paths.code_asset(segmentation["weights_directory"]))
    if paths.raw_inspect_root is not None:
        # Segmentation consumes raw NIfTI files; dataset manifests may instead point
        # at the preprocessed .npy cache used by downstream training stages.
        segmentation["raw_image_root"] = str(
            (paths.raw_inspect_root / "CT" / "full" / "CTPA").resolve()
        )
    lungmask = dict(segmentation.get("lungmask") or {})
    if lungmask.get("checkpoint"):
        lungmask["checkpoint"] = str(paths.code_asset(lungmask["checkpoint"]))
    segmentation["lungmask"] = lungmask
    config["segmentation"] = segmentation
    gpu_ids = (
        parse_devices(args.gpus)
        if args.gpus is not None
        else parse_devices(segmentation.get("devices"))
    )
    config["segmentation"]["devices"] = gpu_ids
    require_preflight(config, paths)
    manager = OutputManager(paths)
    experiment_id = str(config["experiment"]["id"])
    snapshot = {**config, "resolved_paths": paths.as_dict()}
    run_dir, resumed = prepare_resumable_run(
        manager, "segmentation", experiment_id, snapshot, overwrite=args.overwrite
    )
    started = time.perf_counter()
    logger = RunLogger(run_dir / "logs" / "run.log")
    previous_excepthook = sys.excepthook
    sys.excepthook = lambda exc_type, exc, tb: (
        logger.exception("segmentation status=failed", exc),  # noqa: PLE1205 - RunLogger.exception(message, error)
        previous_excepthook(exc_type, exc, tb),
    )[-1]
    logger.log(
        f"segmentation run={experiment_id} status=started resumed={resumed} "
        f"gpus={gpu_ids or ['cpu']}"
    )
    logger.log(f"command={' '.join(sys.argv)}")
    logger.log(f"config={args.config.resolve()} dataset_profile={config['data'].get('profile')} split=all")
    manifest_path = Path(str(config["data"]["manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = paths.dataset_root_for(config) / manifest_path
    source_rows = select_patient_rows(read_rows(manifest_path), args.patient_ids)
    diagnosis_path = paths.dataset_root_for(config) / "manifests" / "diagnosis.csv"
    if diagnosis_path.is_file():
        pe_labels = study_pe_labels(read_rows(diagnosis_path))
        source_rows = [
            {**row, "pe_present": pe_labels.get((str(row["patient_id"]), str(row["study_id"])), row.get("pe_present"))}
            for row in source_rows
        ]
    generated, evaluation = generate_pseudo_anatomy(
        source_rows,
        data_root=paths.dataset_root_for(config),
        image_column=str(config["data"].get("file_column", "image_path")),
        run_dir=run_dir,
        segmentation_config=segmentation,
        maximum_cases=args.max_cases,
        gpu_ids=gpu_ids,
        progress=logger.log,
    )
    write_parquet_atomic(generated, run_dir / "manifest.parquet")
    # One flat row per (study, anatomy): the file to open first when checking whether the
    # masks are plausible. Replaces logs/segmentation_qc.csv, which packed QC into JSON.
    write_csv_atomic([qc_summary_row(row) for row in generated], run_dir / "qc_summary.csv")
    (run_dir / "logs" / "segmentation_qc.csv").unlink(missing_ok=True)
    incomplete = bool(evaluation["studies"]["failed"] or evaluation["studies"]["partially_failed"])
    result = compact_result(
        config,
        status="completed_with_failures" if incomplete else "completed",
        data={**evaluation["studies"], "resumed_existing_run": resumed},
        compute={"strategy": "study_sharding" if len(gpu_ids) > 1 else "single", "gpu_count": len(gpu_ids)},
        evaluation=evaluation,
        reproducibility=environment_report(paths.code_root),
    )
    manager.write_result(run_dir, result, split_artifacts=False)
    logger.log(
        f"requested={evaluation['studies']['requested']} processed={evaluation['studies']['processed']} "
        f"failed={evaluation['studies']['failed']} "
        f"partially_failed={evaluation['studies']['partially_failed']} backend={evaluation['backend']}"
    )
    logger.log(json.dumps(evaluation, indent=2, sort_keys=True))
    logger.log(f"segmentation run={experiment_id} status=finished elapsed_sec={time.perf_counter()-started:.3f}")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
