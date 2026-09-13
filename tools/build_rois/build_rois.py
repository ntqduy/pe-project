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
from source.roi import build_roi_dataset
from source.utils.config import load_config, validate_config
from source.utils.environment import environment_report
from source.utils.logger import RunLogger
from source.utils.qc import qc_row
from tools._common import select_patient_rows, write_csv_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and QC ROI1-ROI8 from pseudo-anatomy masks")
    parser.add_argument("--config", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--patient-id",
        dest="patient_ids",
        action="append",
        metavar="ID",
        help="process every segmented study for this patient; repeat for more patients",
    )
    selection.add_argument("--max-cases", type=int, help="process the first N segmented studies")
    selection.add_argument("--allow-full", action="store_true", help="process the complete segmentation run")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, repeatable; this is how the dataset profile is selected",
    )
    parser.add_argument("--segmentation-run", type=Path, help="existing completed segmentation run")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    config = load_config(args.config, args.overrides)
    config["overwrite"] = bool(args.overwrite)
    config["resume"] = not args.overwrite
    roi_config = dict(config.get("roi") or {})
    if args.segmentation_run is not None:
        roi_config["segmentation_run"] = str(args.segmentation_run)
        config["roi"] = roi_config
        config = validate_config({key: value for key, value in config.items() if key != "config_hash"})
    paths = ProjectPaths.resolve(config)
    segmentation_run = paths.output_asset(roi_config.get("segmentation_run"))
    if segmentation_run is None:
        raise RuntimeError("roi.segmentation_run or --segmentation-run is required")
    if not segmentation_run.is_dir():
        raise SystemExit(
            "ERROR: segmentation run not found:\n"
            f"{segmentation_run}\n\n"
            "ROI construction requires an existing completed segmentation run."
        )
    require_preflight(config, paths)
    manifest_path = segmentation_run / "manifest.parquet"
    manager = OutputManager(paths)
    experiment_id = str(config["experiment"]["id"])
    snapshot = {
        **config,
        "resolved_paths": paths.as_dict(),
        "resolved_segmentation_run": str(segmentation_run),
        "resolved_segmentation_manifest": str(manifest_path),
    }
    run_dir, resumed = prepare_resumable_run(
        manager,
        "roi",
        experiment_id,
        snapshot,
        overwrite=args.overwrite,
    )
    started = time.perf_counter()
    logger = RunLogger(run_dir / "logs" / "run.log")
    previous_excepthook = sys.excepthook
    sys.excepthook = lambda exc_type, exc, tb: (
        logger.exception("roi status=failed", exc),
        previous_excepthook(exc_type, exc, tb),
    )[-1]
    logger.log(
        f"roi run={experiment_id} status=started resumed={resumed} "
        f"workers={int(roi_config.get('workers', 1))}"
    )
    logger.log(f"command={' '.join(sys.argv)}")
    logger.log(
        f"config={args.config.resolve()} dataset_profile={config['data'].get('profile')} "
        f"segmentation_checkpoint={manifest_path.resolve()}"
    )
    segmentation_rows = select_patient_rows(read_rows(manifest_path), args.patient_ids)
    generated, evaluation = build_roi_dataset(
        segmentation_rows,
        run_dir=run_dir,
        roi_config=roi_config,
        maximum_cases=args.max_cases,
        seed=int(config.get("seed", 42)),
        source_segmentation_run=segmentation_run.name,
        source_segmentation_manifest=manifest_path,
        progress=logger.log,
    )
    write_csv_atomic(generated, run_dir / "roi_manifest.csv")
    write_csv_atomic(
        [
            qc_row(
                run_id=experiment_id,
                patient_id=row.get("patient_id"),
                study_id=row.get("study_id"),
                stage="roi",
                item_name=f"{row.get('roi_code', row.get('roi_id'))}_{row.get('roi_name', '')}",
                status=row.get("status"),
                input_path=row.get("image_path"),
                output_path=row.get("mask_path", row.get("roi_path")),
                qc_checks=json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "qc_severity", "voxel_count", "physical_volume_mm3",
                            "geometry_match", "body_contained", "overlap_voxels",
                            "forbidden_overlap_voxels", "dice_with_source",
                            "physical_volume_error_mm3",
                        )
                    },
                    sort_keys=True,
                ),
                failure_reason=row.get("failure_reason", row.get("reason")),
            )
            for row in generated
        ],
        run_dir / "logs" / "roi_qc.csv",
    )
    result = compact_result(
        config,
        status="completed" if not evaluation["studies"]["failed"] else "completed_with_failures",
        data={
            **evaluation["studies"],
            "resumed_existing_run": resumed,
            "source_segmentation_run": segmentation_run.name,
            "source_segmentation_manifest": str(manifest_path),
        },
        compute={"strategy": "cpu_study_sharding", "workers": int(roi_config.get("workers", 1))},
        evaluation=evaluation,
        reproducibility=environment_report(paths.code_root),
    )
    manager.write_result(run_dir, result, split_artifacts=False)
    logger.log(
        f"requested={evaluation['studies']['requested']} processed={evaluation['studies']['processed']} "
        f"failed={evaluation['studies']['failed']}"
    )
    logger.log(json.dumps(evaluation, indent=2, sort_keys=True))
    logger.log(f"roi run={experiment_id} status=finished elapsed_sec={time.perf_counter()-started:.3f}")
    return 0 if not evaluation["studies"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
