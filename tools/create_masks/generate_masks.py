from __future__ import annotations

import argparse
import json
import sys
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
from tools._common import select_patient_rows, write_parquet_atomic


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
    parser.add_argument("--gpus", help="physical GPU IDs for independent study workers, e.g. 0,1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    config = load_config(args.config)
    config["overwrite"] = bool(args.overwrite)
    config["resume"] = not args.overwrite
    paths = ProjectPaths.resolve(config)
    segmentation = dict(config.get("segmentation") or {})
    if segmentation.get("repository"):
        segmentation["repository"] = str(paths.code_asset(segmentation["repository"]))
    if segmentation.get("weights_directory"):
        segmentation["weights_directory"] = str(paths.code_asset(segmentation["weights_directory"]))
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
    manifest_path = Path(str(config["data"]["manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = paths.dataset_root(str(config["data"]["mode"])) / manifest_path
    source_rows = select_patient_rows(read_rows(manifest_path), args.patient_ids)
    generated, evaluation = generate_pseudo_anatomy(
        source_rows,
        data_root=paths.dataset_root(str(config["data"]["mode"])),
        image_column=str(config["data"].get("file_column", "image_path")),
        run_dir=run_dir,
        segmentation_config=segmentation,
        maximum_cases=args.max_cases,
        gpu_ids=gpu_ids,
    )
    write_parquet_atomic(generated, run_dir / "manifest.parquet")
    result = compact_result(
        config,
        status="completed" if not evaluation["studies"]["failed"] else "completed_with_failures",
        data={**evaluation["studies"], "resumed_existing_run": resumed},
        compute={"strategy": "study_sharding" if len(gpu_ids) > 1 else "single", "gpu_count": len(gpu_ids)},
        evaluation=evaluation,
        reproducibility=environment_report(paths.code_root),
    )
    manager.write_result(run_dir, result)
    RunLogger(run_dir / "logs" / "generation.log", echo=False).log(
        f"requested={evaluation['studies']['requested']} processed={evaluation['studies']['processed']} "
        f"failed={evaluation['studies']['failed']} backend={evaluation['backend']}"
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return 0 if not evaluation["studies"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
