#!/usr/bin/env python
"""Run one baseline case: prepare split manifests -> train -> evaluate (+ Grad-CAM preview).

    python tools/baselines/run_case.py --model resnet18_3d                       # official split, MLP, 100%
    python tools/baselines/run_case.py --model vit_3d --head kan --gpus 1
    python tools/baselines/run_case.py --model swin_3d --fraction 25 --fold 2 --folds 5
    python tools/baselines/run_case.py --model convnext_2d --action preflight
    python tools/baselines/run_case.py --model resnet18_3d --task prognosis --target 1_month_mortality

One case = (task, model, head, training fraction, fold, seed, settings). It maps to exactly
one output folder (tools/baselines/experiments.py), so any number of cases can run in parallel
on different GPUs without touching each other. Training flags that differ from the config
(--scratch, --lr, --batch-size, --accumulation, --patience, --no-epoch-auc, extra --set) are
stamped into the folder name as ``__x<settings>``; a case with the config's own settings
keeps the plain name. Stages that already finished are skipped
(tools/run_status.py, the same check scripts/tool/run_ctfm_frozen.sh uses); --overwrite
replaces the case. Everything scientific stays in the run config
(configs/runs/02_diagnosis/baselines/<dim>/<model>.yaml); this script only adds --set overrides.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.baselines.experiments import (  # noqa: E402
    EXPERIMENTS,
    config_path as model_config_path,
    run_tag,
    settings_stamp,
    task_directory,
)

PROGNOSIS_MANIFESTS = {"all": "prognosis_all_patient.csv", "pe": "prognosis_pe_positive.csv"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="config name in configs/runs/02_diagnosis/baselines/<dim>/")
    parser.add_argument("--head", default="mlp", choices=["mlp", "kan"])
    parser.add_argument("--fraction", default="100", help="percent of the training patients (25, 50, 75, 100)")
    parser.add_argument("--fold", default="official", help="official or 0..K-1")
    parser.add_argument("--folds", type=int, default=5, help="K of the k-fold assignment")
    parser.add_argument("--split-seed", type=int, default=42, help="seed of the fold / fraction assignment")
    parser.add_argument("--seed", type=int, default=42, help="training seed")
    parser.add_argument("--profile", default=os.environ.get("PROFILE", "full_inspect"))
    parser.add_argument("--task", default="diagnosis", choices=["diagnosis", "prognosis"])
    parser.add_argument("--target", default="1_month_mortality", help="prognosis endpoint")
    parser.add_argument("--cohort", default="all", choices=sorted(PROGNOSIS_MANIFESTS), help="prognosis cohort")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"), help="'' for CPU, '0', '0,1' (DDP)")
    parser.add_argument("--action", default="all", choices=["all", "prepare", "train", "evaluate", "preflight", "dry"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "auto"))
    parser.add_argument("--no-epoch-auc", action="store_true", help="skip the per-epoch train/val AUROC pass")
    parser.add_argument("--scratch", action="store_true", help="ignore pretrained weights (random init)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variant", default="", help="tag suffix for cases with experiment-specific overrides")
    parser.add_argument("--set", dest="extra", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def fraction_percent(value: str) -> int:
    """--fraction is always a whole percent here ("1" = 1%), matching the grid's run tags."""
    from source.data.experiment_splits import parse_fraction

    text = str(value).strip()
    try:
        percent = parse_fraction(text if text.endswith("%") else f"{text}%") * 100
    except ValueError:
        # parse_fraction's own message would quote the "%"-suffixed text, not what was typed.
        raise SystemExit(f"--fraction must be a whole percent (1-100), got {value!r}") from None
    if abs(percent - round(percent)) > 1e-6:
        raise SystemExit(f"--fraction must be a whole percent (1-100), got {value!r}")
    return int(round(percent))


def base_overrides(args: argparse.Namespace) -> list[str]:
    """Task-level overrides (applied before the manifest is known)."""
    overrides = [f"data.profile={args.profile}"]
    if args.task == "prognosis":
        target = args.target
        overrides += [
            "experiment.stage=prognosis",
            "experiment.family=prognosis",
            f"experiment.id=PR_base_{args.model}",
            f"data.cohort={'all_comers' if args.cohort == 'all' else 'pe_positive_only'}",
            f"data.label_columns=[{target}]",
            f"task.primary_target={target}",
            f"task.targets={{{target}: 1}}",
            "task.modalities=[image]",
        ]
    return overrides


def resolve(args: argparse.Namespace, overrides: list[str]) -> dict:
    from source.utils.config import load_config, validate_config

    config = load_config(model_config_path(args.model), overrides + ["compute.devices=[]"])
    return validate_config({key: value for key, value in config.items() if key != "config_hash"})


def case_settings(args: argparse.Namespace) -> tuple[dict, str]:
    """The case's resolved config (task overrides + user --set) and its ``__x`` settings stamp.

    run_many.py names its launcher logs with the same stamp, so a default grid and a stamped
    grid (e.g. SCRATCH=1) running at the same time never share a log file.
    """
    # User --set values (e.g. paths.*) must already hold when the manifest is located.
    config = resolve(args, base_overrides(args) + list(args.extra))
    # The variant's own overrides (scripts/diagnosis/baselines/<exp>/experiment.yaml) are what the
    # __v<variant> tag already names; only the --set values beyond them are a deviation.
    variant_sets = (EXPERIMENTS.get(args.variant) or {}).get("overrides", {}).get(args.model, []) if args.variant else []
    settings = settings_stamp(
        config,
        scratch=args.scratch,
        lr=args.lr,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        patience=args.patience,
        no_epoch_auc=args.no_epoch_auc,
        extra_sets=[item for item in args.extra if item not in {str(value) for value in variant_sets}],
    )
    return config, settings


def case_manifest(args: argparse.Namespace, config: dict, fraction: int) -> tuple[str, dict | None]:
    """The manifest this case trains on; writes the fold / fraction copy when needed."""
    from source.data.experiment_splits import ExperimentSplits
    from source.data.paths import ProjectPaths

    configured = str(config["data"]["manifest"])
    cached = configured.startswith("manifests/ct_fm/")
    if args.task == "prognosis":
        name = PROGNOSIS_MANIFESTS[args.cohort]
        configured = f"manifests/ct_fm/{name}" if cached else f"manifests/{name}"
    if str(args.fold) == "official" and fraction == 100:
        return configured, None
    base = f"manifests/{PROGNOSIS_MANIFESTS[args.cohort]}" if args.task == "prognosis" else "manifests/diagnosis.csv"
    label = args.target if args.task == "prognosis" else str(config["task"]["primary_target"])
    task_name = f"{args.task}_{args.cohort}" if args.task == "prognosis" else args.task
    root = ProjectPaths.resolve(config).dataset_root_for(config)
    splits = ExperimentSplits(root, task=task_name, label=label, base_manifest=base, folds=args.folds, seed=args.split_seed)
    report = splits.materialize(configured, args.fold, fraction / 100.0)
    return splits.relative_manifest(configured, args.fold, fraction / 100.0), report


def run(command: list[str]) -> int:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    return subprocess.call(command, cwd=ROOT)


def main(argv=None) -> int:
    args = parse_args(argv)
    config_path = model_config_path(args.model)
    fraction = fraction_percent(args.fraction)
    from source.data.experiment_splits import fold_tag

    fold = fold_tag(args.fold)
    overrides = base_overrides(args)
    config, settings = case_settings(args)
    manifest, split_report = case_manifest(args, config, fraction)
    prefix = "PR" if args.task == "prognosis" else "DX"
    task_dir = task_directory(args.task, args.cohort, args.target)
    tag = run_tag(args.model, args.head, fraction, args.variant, settings)
    overrides += [
        f"data.manifest={manifest}",
        f"experiment.id={prefix}_{tag}__{fold}__s{args.seed}",
        f"experiment.output_id=BASE/{args.profile}/{task_dir}/runs/{tag}/{fold}_seed{args.seed}",
        f"head.type={args.head}",
        f"seed={args.seed}",
        f"compute.num_workers={args.num_workers}",
    ]
    if fraction != 100 or fold != "official":
        overrides += [f"data.split_variant={json.dumps({'fold': fold, 'fraction': fraction, 'folds': args.folds, 'split_seed': args.split_seed})}"]
    # Checkpoint / result.json lineage read lineage.fold; record the case's split there too.
    overrides += [f"lineage.fold={fold}", f"lineage.train_fraction={fraction}",
                  f"lineage.cv_folds={args.folds if fold != 'official' else 0}", f"lineage.split_seed={args.split_seed}"]
    for key, value in (("training.epochs", args.epochs), ("training.early_stopping_patience", args.patience),
                       ("training.batch_size", args.batch_size), ("training.gradient_accumulation", args.accumulation),
                       ("training.learning_rate", args.lr)):
        if value is not None:
            overrides.append(f"{key}={value}")
    if args.no_epoch_auc:
        overrides.append("training.record_epoch_auc=false")
    if args.scratch:
        overrides.append("model.pretrained.enabled=false")
    overrides += list(args.extra)
    print(f"==> case task={args.task} model={args.model} head={args.head} fraction={fraction}% fold={fold} "
          f"seed={args.seed} profile={args.profile} gpus={args.gpus or 'cpu'}"
          + (f" settings={settings}" if settings else ""), flush=True)
    if split_report:
        print("    split " + json.dumps({k: str(v) if k == "manifest" else v for k, v in split_report.items()}), flush=True)
    if args.action == "prepare":
        return 0

    common = ["--config", str(config_path.relative_to(ROOT)), "--gpus", args.gpus]
    for override in overrides:
        common += ["--set", override]
    if args.action == "preflight":
        return run([args.python, "tools/preflight.py", *common])
    if args.action == "dry":
        return run([args.python, "tools/launch.py", *common, "--dry-run"])

    status = subprocess.run([args.python, "tools/run_status.py", *common, "--format", "json"],
                            cwd=ROOT, capture_output=True, text=True)
    if status.returncode != 0:
        sys.stderr.write(status.stderr)
        return status.returncode
    # JSON, not the text line: the run directory may contain spaces.
    report = json.loads(status.stdout.strip().splitlines()[-1])
    state, run_dir, difference = str(report["state"]), str(report["run_dir"]), list(report.get("different") or [])
    if state == "different" and not args.overwrite:
        raise SystemExit(f"{run_dir} holds a completed run with other settings ({','.join(difference)}); use --overwrite")
    if state == "incomplete" and not args.overwrite and args.action in {"all", "train"}:
        raise SystemExit(f"{run_dir} holds an unfinished run; use --overwrite to restart it")
    overwrite = ["--overwrite"] if args.overwrite else []
    if args.action in {"all", "train"}:
        if state in {"trained", "evaluated"} and not args.overwrite:
            print(f"==> train: skipped, {run_dir} already holds a completed run", flush=True)
        else:
            code = run([args.python, "tools/launch.py", "--quiet", *common, *overwrite])
            if code:
                return code
            state = "trained"
    if args.action in {"all", "evaluate"}:
        if state == "evaluated" and not args.overwrite:
            print(f"==> evaluate: skipped, {run_dir}/result.csv already exists", flush=True)
        else:
            code = run([args.python, "tools/launch.py", "--quiet", "--evaluate", "--allow-full", *common, *overwrite])
            if code:
                return code
    print(f"==> done: {run_dir}", flush=True)
    for name in ("result.csv", "predictions.csv", "training_curves.png", "preview", "logs.txt"):
        print(f"    {name}: {Path(run_dir) / name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
