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
from source.segmentation.pipeline import generate_pseudo_anatomy
from source.utils.config import load_config, parse_devices
from source.utils.environment import environment_report
from source.utils.logger import RunLogger
from source.utils.qc import qc_row
from tools._common import select_patient_rows, write_csv_atomic, write_parquet_atomic


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
        logger.exception("segmentation status=failed", exc),
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
    write_csv_atomic(
        [
            qc_row(
                run_id=experiment_id,
                patient_id=row.get("patient_id"),
                study_id=row.get("study_id"),
                stage="segmentation",
                item_name=row.get("anatomy"),
                status=row.get("status"),
                input_path=row.get("image_path"),
                output_path=row.get("mask_path"),
                qc_checks=json.dumps(
                    {key: row.get(key) for key in ("status", "reason", "volume_ml", "cross_model_dice")},
                    sort_keys=True,
                ),
                failure_reason=row.get("reason") if str(row.get("status")) in {"FAIL", "UNAVAILABLE"} else "",
            )
            for row in generated
        ],
        run_dir / "logs" / "segmentation_qc.csv",
    )
    result = compact_result(
        config,
        status="completed" if not evaluation["studies"]["failed"] else "completed_with_failures",
        data={**evaluation["studies"], "resumed_existing_run": resumed},
        compute={"strategy": "study_sharding" if len(gpu_ids) > 1 else "single", "gpu_count": len(gpu_ids)},
        evaluation=evaluation,
        reproducibility=environment_report(paths.code_root),
    )
    manager.write_result(run_dir, result, split_artifacts=False)
    logger.log(
        f"requested={evaluation['studies']['requested']} processed={evaluation['studies']['processed']} "
        f"failed={evaluation['studies']['failed']} backend={evaluation['backend']}"
    )
    logger.log(json.dumps(evaluation, indent=2, sort_keys=True))
    logger.log(f"segmentation run={experiment_id} status=finished elapsed_sec={time.perf_counter()-started:.3f}")
    return 0 if not evaluation["studies"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
