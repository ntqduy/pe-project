from __future__ import annotations

import shlex
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.launcher import build_launch_spec, launch
from source.utils.config import infer_compute_strategy
from tools._common import base_parser, resolve_cli_config

ENTRYPOINTS = {
    "foundation": "tools/pretrain_model/materialize_foundation.py",
    "dapt": "tools/pretrain_model/train_dapt.py",
    "alignment": "tools/pretrain_model/train_alignment.py",
    "silver": "tools/silver_labels/generate_silver_labels.py",
    "silver_encoder_adaptation": "tools/pretrain_model/train_silver_encoder.py",
    "diagnosis": "tools/tasks/train_task.py",
    "prognosis": "tools/tasks/train_task.py",
    "contour": "tools/tasks/train_task.py",
    "roi_student": "tools/tasks/train_task.py",
    "counterfactual": "tools/tasks/counterfactual.py",
}


def main() -> int:
    parser = base_parser(
        "Launch one training or silver-generation experiment on CPU, one GPU, or PyTorch DDP"
    )
    parser.add_argument("--dry-run", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--patient-id", dest="patient_ids", action="append", metavar="ID")
    selection.add_argument("--max-reports", type=int)
    selection.add_argument("--max-cases", type=int)
    selection.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    config = resolve_cli_config(args)
    stage = str(config["experiment"]["stage"])
    if stage == "ablation":
        stage = str((config.get("task") or {}).get("base_stage") or "")
    if stage not in ENTRYPOINTS:
        raise SystemExit(f"stage {stage!r} is not supported by the launcher")
    selected_scope = bool(
        args.patient_ids or args.max_reports is not None or args.max_cases is not None or args.allow_full
    )
    silver_selection = bool(args.patient_ids or args.max_reports is not None or args.allow_full)
    if stage == "silver" and not silver_selection:
        raise SystemExit("silver generation requires --patient-id, --max-reports, or --allow-full")
    if stage == "counterfactual" and not selected_scope:
        raise SystemExit("counterfactual inference requires --patient-id, --max-cases, or --allow-full")
    if stage not in {"silver", "counterfactual"} and selected_scope:
        raise SystemExit("patient/item selection is only valid for silver or counterfactual inference")
    if stage == "silver" and args.max_cases is not None:
        raise SystemExit("silver generation uses --max-reports, not --max-cases")
    if stage == "counterfactual" and args.max_reports is not None:
        raise SystemExit("counterfactual inference uses --max-cases, not --max-reports")
    if args.max_reports is not None and args.max_reports < 1:
        raise SystemExit("--max-reports must be positive")
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    if stage == "counterfactual" and not args.allow_full:
        config["resume"] = True
    if stage == "silver" and not args.overwrite:
        config["resume"] = True
    if str(config["experiment"].get("family") or "") == "remove_roi":
        raise SystemExit(
            "remove_roi ablation configs are frozen-checkpoint, no-retraining counterfactual "
            "evaluations (section 10.B); run them with tools/tasks/evaluate.py --checkpoint "
            "<frozen source checkpoint>, not tools/launch.py, which would fine-tune a new model "
            "instead of evaluating the same frozen one."
        )
    paths = ProjectPaths.resolve(config)
    require_preflight(config, paths)
    physical = tuple(config["compute"].get("devices") or ())
    if stage == "foundation" and len(physical) > 1:
        raise SystemExit("foundation materialization supports CPU or one GPU; use one device")
    child_arguments = ["--config", str(args.config)]
    for override in args.overrides:
        child_arguments += ["--set", override]
    logical = list(range(len(physical)))
    child_arguments += ["--set", f"compute.devices={logical}"]
    child_arguments += ["--set", f"compute.strategy={infer_compute_strategy(logical)}"]
    child_arguments += ["--set", f"compute.accelerator={'cuda' if logical else 'cpu'}"]
    if args.resume:
        child_arguments.append("--resume")
    if args.overwrite:
        child_arguments.append("--overwrite")
    if stage == "silver":
        if args.patient_ids:
            for patient_id in args.patient_ids:
                child_arguments += ["--patient-id", patient_id]
        elif args.max_reports is not None:
            child_arguments += ["--max-reports", str(args.max_reports)]
        else:
            child_arguments.append("--allow-full")
    elif stage == "counterfactual":
        if args.patient_ids:
            for patient_id in args.patient_ids:
                child_arguments += ["--patient-id", patient_id]
        elif args.max_cases is not None:
            child_arguments += ["--max-cases", str(args.max_cases)]
        else:
            child_arguments.append("--allow-full")
    entrypoint = paths.code_root / ENTRYPOINTS[stage]
    spec = build_launch_spec(entrypoint, child_arguments, physical)
    distributed_mode = "distributed report sharding" if stage == "silver" else "DDP (one experiment)"
    mode = "CPU" if not physical else "single GPU" if len(physical) == 1 else distributed_mode
    print("Launch:", shlex.join(spec.command))
    print("Mode:", mode)
    print("Physical GPUs:", ",".join(map(str, physical)) if physical else "none")
    if args.dry_run:
        return 0
    process = launch(spec)
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
