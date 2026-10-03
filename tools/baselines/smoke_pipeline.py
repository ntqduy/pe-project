#!/usr/bin/env python
"""End-to-end smoke check of the baseline pipeline on synthetic data (no real training).

    python tools/baselines/smoke_pipeline.py                     # default grid, 2 cases at a time
    python tools/baselines/smoke_pipeline.py --models resnet18_3d --tasks diagnosis --seeds 0 1
    python tools/baselines/smoke_pipeline.py --jobs 1 --skip-cases   # only re-run the checks

Every case is ``run_case.py --smoke`` (tools/baselines/smoke_data.py: synthetic volumes, all
paths under $PE_SMOKE_ROOT or output/_smoke). Checks, each reported PASS / FAIL in
``<smoke root>/smoke_report.md``:

  cases          train -> validation-AUROC checkpoint -> evaluate for every model x head x task x
                 seed; result.csv has every metric column; predictions.csv covers train /
                 validation / test; visualize/correct + visualize/incorrect hold Grad-CAMs
  label rule     --task prognosis without --label, --task diagnosis with --label and an unknown
                 label are rejected before anything runs
  reproducible   the first case re-run with the same seed gives the same probabilities
  early stop     Trainer stops after `patience` epochs without a better validation AUROC
  aggregate      summarize.py on >= 2 seeds; the seed ensemble is unchanged when predictions.csv
                 rows are shuffled, and refuses runs whose studies differ
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Empty, Queue

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.baselines.experiments import run_tag, task_directory  # noqa: E402
from tools.baselines.smoke_data import PROFILE, default_root, write_dataset  # noqa: E402

RESULT_COLUMNS = ("auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high", "threshold",
                  "threshold_rule", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy",
                  "accuracy", "brier", "note")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["resnet18_2d", "resnet18_25d", "resnet18_3d"])
    parser.add_argument("--heads", nargs="+", default=["mlp", "kan"])
    parser.add_argument("--tasks", nargs="+", default=["diagnosis", "prognosis"], choices=["diagnosis", "prognosis"])
    parser.add_argument("--label", default="1_month_mortality", help="prognosis label of the smoke cases")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs", type=int, default=2, help="cases run at the same time")
    parser.add_argument("--smoke-root", type=Path, default=default_root())
    parser.add_argument("--skip-cases", action="store_true", help="reuse finished cases, only run the checks")
    parser.add_argument("--cases-only", action="store_true",
                        help="only the per-case checks (e.g. one seed of every arm); skip label / repeat / aggregate")
    parser.add_argument("--report-name", default="smoke_report.md")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="keep best/last.ckpt of smoke runs (default: deleted once a case is checked; "
                             "they are 0.1-1 GB each and mean nothing)")
    return parser.parse_args(argv)


class Report:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
        return bool(ok)


def case_dir(root: Path, case: dict) -> Path:
    task_dir = task_directory(case["task"], "all", case.get("label"))
    family = "diagnosis" if case["task"] == "diagnosis" else "prognosis"
    return (root / "pe-project" / "outputs" / family / "BASE" / PROFILE / task_dir / "runs"
            / run_tag(case["model"], case["head"], 100) / f"official_seed{case['seed']}" / "epoch_2")


def case_command(args: argparse.Namespace, case: dict, *extra: str) -> list[str]:
    command = [sys.executable, "tools/baselines/run_case.py", "--smoke", "--smoke-root", str(args.smoke_root),
               "--model", case["model"], "--head", case["head"], "--seed", str(case["seed"]), "--task", case["task"]]
    if case.get("label"):
        command += ["--label", case["label"]]
    return command + list(extra)


def run_cases(args: argparse.Namespace, cases: list[dict], log_dir: Path) -> dict[str, tuple[int, float]]:
    queue: Queue = Queue()
    for case in cases:
        queue.put(case)
    outcome: dict[str, tuple[int, float]] = {}
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                case = queue.get_nowait()
            except Empty:
                return
            name = case["name"]
            started = time.time()
            with (log_dir / f"{name}.log").open("w", encoding="utf-8") as handle:
                code = subprocess.call(case_command(args, case, "--overwrite"), cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT)
            if not args.keep_checkpoints:
                prune_checkpoints(case_dir(args.smoke_root, case))
            with lock:
                outcome[name] = (code, time.time() - started)
                print(f"[{'done' if code == 0 else 'FAIL'}] {name} {(time.time() - started) / 60:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.jobs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcome


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prune_checkpoints(run: Path) -> None:
    """Smoke weights are worthless and large; drop them once a case has been checked."""
    for path in (*run.glob("checkpoint/*.ckpt"), *run.glob("*.ckpt")):
        path.unlink(missing_ok=True)


def check_case(report: Report, case: dict, run: Path, code: int | None) -> dict:
    name = case["name"]
    row = {"case": name, "status": "ok" if code == 0 else f"exit {code}"}
    if code not in (0, None):
        report.check(f"case {name}", False, f"run_case exited {code}; see its log")
        return row
    problems = []
    result = run / "result.csv"
    predictions = run / "predictions.csv"
    if not result.is_file():
        problems.append("no result.csv")
    else:
        table = read_rows(result)
        missing = [column for column in RESULT_COLUMNS if column not in table[0]]
        if missing:
            problems.append(f"result.csv lacks {missing}")
        test = next((item for item in table if item["split"] == "test"), {})
        row["test_auroc"] = test.get("auroc", "")
        row["auroc_ci"] = f"[{test.get('auroc_ci_low', '')}, {test.get('auroc_ci_high', '')}]"
    if not predictions.is_file():
        problems.append("no predictions.csv")
    else:
        splits = {item["split"] for item in read_rows(predictions)}
        if splits != {"train", "validation", "test"}:
            problems.append(f"predictions.csv splits {sorted(splits)}")
    correct = len(list((run / "visualize" / "correct").glob("*.png")))
    incorrect = len(list((run / "visualize" / "incorrect").glob("*.png")))
    row["visualize"] = f"{correct} correct / {incorrect} incorrect"
    if correct + incorrect == 0:
        problems.append("visualize/ holds no Grad-CAM")
    if case["model"].endswith(("_2d", "_25d")) and not list((run / "visualize").rglob("*_mil_attention.png")):
        problems.append("slice-MIL run without _mil_attention.png")
    if not (run / "training_curves.png").is_file():
        problems.append("no training_curves.png")
    payload = json.loads((run / "result.json").read_text(encoding="utf-8")) if (run / "result.json").is_file() else {}
    training = (payload.get("evaluation") or {}).get("training") or {}
    row["selection"] = training.get("selection_metric", "")
    if training.get("selection_metric") != "validation_auroc":
        problems.append(f"selection metric {training.get('selection_metric')!r}")
    report.check(f"case {name}", not problems, "; ".join(problems))
    return row


def check_labels(report: Report, args: argparse.Namespace) -> None:
    probes = [
        ("prognosis without --label", ["--task", "prognosis"], "requires --label"),
        ("diagnosis with --label", ["--task", "diagnosis", "--label", "1_month_mortality"], "only valid with --task prognosis"),
        ("unknown --label", ["--task", "prognosis", "--label", "30_day_mortality"], "invalid choice"),
    ]
    for title, flags, expected in probes:
        command = [sys.executable, "tools/baselines/run_case.py", "--model", "resnet18_3d", "--action", "dry", *flags]
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        text = done.stdout + done.stderr
        report.check(f"label rule: {title}", done.returncode != 0 and expected in text,
                     f"exit {done.returncode}: {text.strip().splitlines()[-1] if text.strip() else ''}")


def check_reproducible(report: Report, args: argparse.Namespace, case: dict, run: Path) -> None:
    before = {(r["split"], r["study_id"]): float(r["y_prob"]) for r in read_rows(run / "predictions.csv")}
    code = subprocess.call(case_command(args, case, "--overwrite"), cwd=ROOT, stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)
    if code:
        report.check("reproducible: same seed twice", False, f"re-run exited {code}")
        return
    after = {(r["split"], r["study_id"]): float(r["y_prob"]) for r in read_rows(run / "predictions.csv")}
    same_keys = set(before) == set(after)
    delta = max((abs(before[key] - after[key]) for key in before if key in after), default=float("inf"))
    report.check("reproducible: same seed twice", same_keys and delta <= 1e-6,
                 f"{case['name']}: max |dp| = {delta:.2e} over {len(before)} predictions")
    if not args.keep_checkpoints:
        prune_checkpoints(run)


def check_early_stopping(report: Report) -> None:
    """Trainer on a toy model with a flat validation AUROC must stop after `patience` epochs."""
    import torch

    from source.distributed.setup import DistributedContext
    from source.engine import trainer as trainer_module
    from source.engine.trainer import Trainer

    model = torch.nn.Linear(2, 1)
    data = [{"x": torch.randn(4, 2), "y": torch.rand(4, 1)} for _ in range(2)]

    def loss_step(module, batch):
        return torch.nn.functional.mse_loss(module(batch["x"]), batch["y"])

    # Only the stopping rule is under test: checkpoint writing (and its lineage contract, which a
    # toy model cannot meet) is replaced by a marker file for the duration of the check.
    saved = trainer_module.save_checkpoint_atomic
    trainer_module.save_checkpoint_atomic = lambda path, *_, **__: Path(path).write_bytes(b"smoke")
    try:
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), loss_step, DistributedContext(),
                              Path(folder), {}, early_stopping_patience=2, selection_metric="val_auroc")
            result = trainer.fit(data, data, epochs=10, epoch_metrics_fn=lambda *_: {"val_auroc": 0.5})
            ok = result["stopped_early"] and result["epochs_run"] == 3 and (Path(folder) / "best.ckpt").is_file()
            trainer.logger = None
    finally:
        trainer_module.save_checkpoint_atomic = saved
    report.check("early stopping on flat validation AUROC", ok,
                 f"patience 2, budget 10: ran {result['epochs_run']} epochs, stopped_early={result['stopped_early']}")


def check_aggregate(report: Report, args: argparse.Namespace, cases: list[dict]) -> None:
    from tools.baselines import summarize

    outputs = args.smoke_root / "pe-project" / "outputs"
    for task in args.tasks:
        label = args.label if task == "prognosis" else None
        family = "diagnosis" if task == "diagnosis" else "prognosis"
        base = outputs / family / "BASE" / PROFILE / task_directory(task, "all", label)
        for exp in ("exp01_baselines", "exp03_head_ablation"):
            command = [sys.executable, "tools/baselines/summarize.py", "--exp", exp, "--task", task, "--base", str(base)]
            if label:
                command += ["--label", label]
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            pretty = base / exp / "summary_pretty.csv"
            rows = read_rows(pretty) if pretty.is_file() else []
            multi = [row for row in rows if int(row.get("n_seeds") or 0) >= 2 and "[" in row.get("auroc_[CI]_seed_avg", "")]
            report.check(f"aggregate {task} {exp}", done.returncode == 0 and bool(multi),
                         f"{len(rows)} model rows, {len(multi)} with >= 2 seeds and a seed-ensemble CI"
                         + ("" if done.returncode == 0 else f"; {done.stderr.strip().splitlines()[-1:]}"))
    # Seed ensemble: merged on study_id, so shuffled rows change nothing; differing studies fail.
    group = [case for case in cases if case["task"] == args.tasks[0] and case["model"] == args.models[-1]
             and case["head"] == args.heads[0]]
    items = [{"run_dir": str(case_dir(args.smoke_root, case)), "seed": case["seed"], "primary_target": ""} for case in group]
    if len(items) < 2:
        report.check("seed ensemble merge", False, "needs >= 2 seeds of one model")
        return
    stage = args.tasks[0]
    original = summarize.seed_ensemble(items, stage)
    path = Path(items[-1]["run_dir"]) / "predictions.csv"
    backup = path.read_bytes()
    try:
        rows = read_rows(path)
        random.Random(0).shuffle(rows)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        shuffled = summarize.seed_ensemble(items, stage)
        keys = ("auroc", "auprc", "threshold", "sensitivity", "specificity", "brier")
        report.check("seed ensemble ignores row order", all(original[k] == shuffled[k] for k in keys),
                     f"AUROC {original['auroc']} vs {shuffled['auroc']} after shuffling {path.parent.parent.name}")
        first_test = next(index for index, row in enumerate(rows) if row["split"] == "test")
        test_rows = rows[:first_test] + rows[first_test + 1:]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(test_rows)
        try:
            summarize.seed_ensemble(items, stage)
            report.check("seed ensemble rejects differing studies", False, "a run missing one test study was accepted")
        except SystemExit as error:
            report.check("seed ensemble rejects differing studies", "differ" in str(error), str(error)[:140])
    finally:
        path.write_bytes(backup)


def write_report(args: argparse.Namespace, report: Report, rows: list[dict]) -> Path:
    lines = ["# Baseline pipeline smoke report", "",
             f"Synthetic data, CPU/GPU as available; smoke root `{args.smoke_root}`. Numbers mean nothing; "
             "only PASS/FAIL matters.", "", "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in report.checks]
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        lines += ["", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        lines += ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    path = args.smoke_root / args.report_name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    args = parse_args(argv)
    write_dataset(args.smoke_root)
    log_dir = args.smoke_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for model, head, task, seed in itertools.product(args.models, args.heads, args.tasks, args.seeds):
        label = args.label if task == "prognosis" else None
        cases.append({"model": model, "head": head, "task": task, "label": label, "seed": seed,
                      "name": f"{task}{'_' + label if label else ''}__{model}__{head}__s{seed}"})
    report = Report()
    started = time.time()
    outcome = {} if args.skip_cases else run_cases(args, cases, log_dir)
    rows = [check_case(report, case, case_dir(args.smoke_root, case), outcome.get(case["name"], (None, 0))[0])
            for case in cases]
    if not args.keep_checkpoints:
        for case in cases:
            prune_checkpoints(case_dir(args.smoke_root, case))
    checks = () if args.cases_only else (
        ("label rule", lambda: check_labels(report, args)),
        ("reproducible", lambda: check_reproducible(report, args, cases[0], case_dir(args.smoke_root, cases[0]))),
        ("early stopping", lambda: check_early_stopping(report)),
        ("aggregate", lambda: check_aggregate(report, args, cases)),
    )
    for title, check in checks:
        try:
            check()
        except Exception as error:  # noqa: BLE001 - a broken check is a FAIL in the report, not a crash
            report.check(f"{title} (check crashed)", False, f"{type(error).__name__}: {error}"[:300])
    path = write_report(args, report, rows)
    failed = [name for name, ok, _ in report.checks if not ok]
    print(f"==> {len(report.checks) - len(failed)}/{len(report.checks)} checks passed in "
          f"{(time.time() - started) / 60:.1f} min; report: {path}; case logs: {log_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
