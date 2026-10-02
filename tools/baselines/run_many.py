#!/usr/bin/env python
"""Run a grid of baseline cases, in parallel over GPU slots, then summarize.

    # one experiment, everything it needs, 4 GPUs, one case per GPU at a time
    python tools/baselines/run_many.py --exp exp01_baselines --gpus 0,1,2,3

    # a subset / another grid; k-fold (official split is 'official')
    python tools/baselines/run_many.py --exp exp02_data_fraction --models vit_3d swin_3d --fractions 25 50
    python tools/baselines/run_many.py --exp exp03_head_ablation --folds-to-run 0 1 2 3 4 --gpus 0,1

    # two small cases per GPU (e.g. CT-FM frozen heads), or one DDP case over two GPUs
    python tools/baselines/run_many.py --exp exp03_head_ablation --models ctfm_frozen_3d --jobs-per-gpu 2
    python tools/baselines/run_many.py --exp exp01_baselines --models vmamba_3d --gpus-per-job 2 --gpus 0,1

Each case is tools/baselines/run_case.py in its own process with its own GPU set and its own
log file (<outputs>/<family>/BASE/<profile>/<task>/<exp>/launcher_logs/<run tag>__<fold>__s<seed>.log, the run tag
carrying the same __x<settings> stamp as the case's output folder); finished cases are skipped, so re-running
the same command resumes a grid. A failed case does not stop the others; the exit code is
non-zero if any failed. --dry-list prints the grid without running it.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.baselines.experiments import (  # noqa: E402
    EXPERIMENTS, PROGNOSIS_LABELS, base_directory, check_label, dimension_folder, run_tag, task_directory,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--models", nargs="+", help="subset of the experiment's models")
    parser.add_argument("--dims", nargs="+", choices=["2D", "2_5D", "3D"], help="only the experiment's models of these dimensions")
    parser.add_argument("--heads", nargs="+", help="override the experiment's heads")
    parser.add_argument("--fractions", nargs="+", type=int, help="override the experiment's fractions")
    parser.add_argument("--folds-to-run", nargs="+", default=["official"], help="official and/or 0..K-1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"), help="GPU pool, e.g. 0,1,2,3; '' = CPU")
    parser.add_argument("--gpus-per-job", type=int, default=1)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--profile", default=os.environ.get("PROFILE", "full_inspect"))
    parser.add_argument("--task", default="diagnosis", choices=["diagnosis", "prognosis"])
    parser.add_argument("--label", default=None, choices=PROGNOSIS_LABELS,
                        help="prognosis endpoint; required with --task prognosis, not allowed with diagnosis")
    parser.add_argument("--cohort", default="all", choices=["all", "pe"], help="prognosis cohort")
    parser.add_argument("--dry-list", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--log-dir", type=Path, help="per-case launcher logs (default: <exp summary dir>/launcher_logs)")
    parser.add_argument("case_args", nargs=argparse.REMAINDER, help="after --, passed to every run_case.py")
    args = parser.parse_args(argv)
    check_label(args.task, args.label)
    return args


def task_arguments(args: argparse.Namespace) -> list[str]:
    """--task / --label / --cohort as run_case.py and summarize.py take them."""
    return ["--task", args.task, *(["--label", args.label] if args.label else []), "--cohort", args.cohort]


def slots(pool: str, per_job: int, per_gpu: int) -> list[str]:
    devices = [device.strip() for device in pool.split(",") if device.strip()]
    if not devices:
        return [""] * max(1, per_gpu)
    groups = [",".join(devices[index : index + per_job]) for index in range(0, len(devices) - per_job + 1, per_job)]
    if not groups:
        raise SystemExit(f"--gpus-per-job {per_job} exceeds the pool {pool}")
    idle = devices[len(groups) * per_job:]
    if idle:
        # e.g. --gpus 0,1,2 --gpus-per-job 2: GPU 2 cannot form a full group and would sit idle.
        print(f"warning: --gpus-per-job {per_job} does not divide the pool {pool}; "
              f"GPU(s) {','.join(idle)} will stay idle", file=sys.stderr, flush=True)
    return [group for group in groups for _ in range(max(1, per_gpu))]


def case_command(args: argparse.Namespace, spec: dict, case: dict, extra: list[str], gpu: str = "") -> list[str]:
    """run_case.py arguments of one case (without the interpreter and script)."""
    command = ["--model", case["model"], "--head", case["head"], "--fraction", str(case["fraction"]),
               "--fold", str(case["fold"]), "--folds", str(args.folds), "--seed", str(case["seed"]),
               "--profile", args.profile, *task_arguments(args), "--gpus", gpu, *extra]
    overrides = spec["overrides"].get(case["model"]) or []
    if overrides:
        command += ["--variant", args.exp] + [part for value in overrides for part in ("--set", str(value))]
    return command


def case_names(args: argparse.Namespace, spec: dict, cases: list[dict], extra: list[str]) -> list[str]:
    """``<run tag>__<fold>__s<seed>`` per case, with the same ``__x<settings>`` stamp run_case.py
    puts in the output folder (run_case.case_settings), so grids with different training flags
    never share a launcher log."""
    from tools.baselines import run_case

    stamps: dict[str, str] = {}
    names = []
    for case in cases:
        parsed = run_case.parse_args(case_command(args, spec, case, extra))
        # Within one grid the stamp depends only on the model (its config and overrides).
        if case["model"] not in stamps:
            try:
                stamps[case["model"]] = run_case.case_settings(parsed)[1]
            except (Exception, SystemExit) as error:  # the case itself reports it when it runs
                print(f"warning: cannot resolve the settings of {case['model']} ({error}); "
                      "its launcher log name is unstamped", file=sys.stderr, flush=True)
                stamps[case["model"]] = ""
        tag = run_tag(case["model"], case["head"], case["fraction"], parsed.variant, stamps[case["model"]])
        names.append(f"{tag}__{case['fold']}__s{case['seed']}")
    return names


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = EXPERIMENTS[args.exp]
    models = args.models or spec["models"]
    if args.dims:
        models = [model for model in models if dimension_folder(model) in set(args.dims)]
        if not models:
            raise SystemExit(f"{args.exp} has no {args.dims} models")
    unknown = sorted(set(models) - set(spec["models"])) if args.models else []
    if unknown:
        print(f"note: {unknown} are not part of {args.exp}; running them anyway", flush=True)
    heads = args.heads or spec["heads"]
    fractions = args.fractions or spec["fractions"]
    extra = [part for part in args.case_args if part != "--"]
    cases = [
        {"model": model, "head": head, "fraction": fraction, "fold": fold, "seed": seed}
        for model, head, fraction, fold, seed in itertools.product(models, heads, fractions, args.folds_to_run, args.seeds)
    ]
    task_dir = task_directory(args.task, args.cohort, args.label)
    task_args = task_arguments(args)
    names = case_names(args, spec, cases, extra)
    log_dir = args.log_dir or base_directory(args.profile, task_dir) / args.exp / "launcher_logs"
    pool = slots(args.gpus, args.gpus_per_job, args.jobs_per_gpu)
    print(f"==> {args.exp}: {spec['title']} | {len(cases)} case(s) | slots={pool}")
    for case, name in zip(cases, names):
        overrides = spec["overrides"].get(case["model"])
        print(f"    {name}" + (f"  overrides={overrides}" if overrides else ""))
    if args.dry_list:
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    queue: Queue = Queue()
    for item in zip(names, cases):
        queue.put(item)
    failures: list[str] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                name, case = queue.get_nowait()
            except Empty:
                return
            command = [sys.executable, "tools/baselines/run_case.py", *case_command(args, spec, case, extra, gpu)]
            started = time.time()
            with lock:
                print(f"[start] {name} gpus={gpu or 'cpu'} log={log_dir / (name + '.log')}", flush=True)
            with (log_dir / f"{name}.log").open("w", encoding="utf-8") as handle:
                code = subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            with lock:
                status = "ok" if code == 0 else f"FAILED (exit {code})"
                print(f"[{'done' if code == 0 else 'fail'}] {name} {status} in {(time.time() - started) / 60:.1f} min", flush=True)
                if code:
                    failures.append(name)

    threads = [threading.Thread(target=worker, args=(slot,), daemon=True) for slot in pool]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not args.no_summary:
        subprocess.call([sys.executable, "tools/baselines/summarize.py", "--exp", args.exp, "--profile", args.profile,
                         *task_args], cwd=ROOT)
    if failures:
        print(f"==> {len(failures)} case(s) failed: {failures} (see {log_dir})", flush=True)
        return 1
    print(f"==> all {len(cases)} case(s) finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
