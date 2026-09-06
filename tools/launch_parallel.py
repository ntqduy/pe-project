from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.data.paths import ProjectPaths, discover_code_root
from source.data.preflight import require_preflight
from source.utils.config import infer_compute_strategy, load_config, parse_devices


@dataclass(frozen=True)
class Job:
    experiment: str
    config_path: Path
    gpus: tuple[int, ...]
    wave: int
    arguments: tuple[str, ...]
    command: tuple[str, ...]


def _read_parallel_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parallel = payload.get("parallel")
    if not isinstance(parallel, dict):
        raise SystemExit("parallel config requires a 'parallel' mapping")
    if not parallel.get("enabled", False):
        raise SystemExit("parallel.enabled is false; enable it explicitly before launching")
    return parallel


def _validate_job_arguments(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise SystemExit("parallel job args must be a list of strings")
    arguments = tuple(raw)
    value_options = {"--patient-id", "--max-reports"}
    flag_options = {"--allow-full", "--resume"}
    selection_modes: set[str] = set()
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if option in flag_options:
            if option == "--allow-full":
                selection_modes.add(option)
            index += 1
            continue
        has_value = index + 1 < len(arguments) and not arguments[index + 1].startswith("--")
        if option in value_options and has_value:
            selection_modes.add(option)
            if option == "--max-reports":
                try:
                    maximum = int(arguments[index + 1])
                except ValueError as exc:
                    raise SystemExit("--max-reports requires a positive integer") from exc
                if maximum < 1:
                    raise SystemExit("--max-reports requires a positive integer")
            index += 2
            continue
        raise SystemExit(f"unsupported or incomplete parallel job argument: {option}")
    if len(selection_modes) > 1:
        raise SystemExit("parallel silver job must use exactly one selection mode")
    return arguments


def _job_requests(raw_jobs: Any, config_path: Path) -> list[tuple[Path, tuple[str, ...]]]:
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise SystemExit("parallel.jobs must be a non-empty list")
    requests: list[tuple[Path, tuple[str, ...]]] = []
    for raw in raw_jobs:
        value = raw.get("config") if isinstance(raw, dict) else raw
        if not value:
            raise SystemExit("every parallel job requires a config path")
        path = Path(str(value))
        resolved = path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
        arguments = _validate_job_arguments(raw.get("args") if isinstance(raw, dict) else None)
        requests.append((resolved, arguments))
    return requests


def _build_jobs(config_path: Path, parallel: dict[str, Any], passthrough: list[str]) -> list[Job]:
    devices = tuple(parse_devices(parallel.get("devices")))
    if not devices:
        raise SystemExit("parallel.devices must list at least one physical GPU")
    gpus_per_job = int(parallel.get("gpus_per_job", 1))
    if gpus_per_job < 1 or gpus_per_job > len(devices):
        raise SystemExit("parallel.gpus_per_job must be between 1 and len(parallel.devices)")
    groups = [devices[index : index + gpus_per_job] for index in range(0, len(devices), gpus_per_job)]
    groups = [group for group in groups if len(group) == gpus_per_job]
    if not groups:
        raise SystemExit("parallel.devices cannot form a complete GPU group")

    claimed_runs: set[tuple[str, str]] = set()
    jobs: list[Job] = []
    for index, (experiment_config, job_arguments) in enumerate(
        _job_requests(parallel.get("jobs"), config_path)
    ):
        group = groups[index % len(groups)]
        wave = index // len(groups)
        overrides = [
            f"compute.devices={list(group)}",
            f"compute.strategy={infer_compute_strategy(group)}",
        ]
        config = load_config(experiment_config, overrides)
        experiment = str(config["experiment"]["id"])
        stage = str(config["experiment"]["stage"])
        selection_options = {"--patient-id", "--max-reports", "--allow-full"}
        has_selection = any(argument in selection_options for argument in job_arguments)
        if stage == "silver" and not has_selection:
            raise SystemExit(f"silver job {experiment} requires a patient/report selection argument")
        if stage != "silver" and has_selection:
            raise SystemExit(f"report selection arguments are invalid for non-silver job {experiment}")
        if "--resume" in job_arguments and "--overwrite" in passthrough:
            raise SystemExit(f"job {experiment} cannot combine --resume with global --overwrite")
        if stage == "foundation" and len(group) > 1:
            raise SystemExit(f"foundation job {experiment} supports at most one GPU")
        family = str(config["experiment"].get("family") or config["experiment"]["stage"])
        if family == "remove_roi":
            raise SystemExit(
                f"job {experiment}: remove_roi ablation configs are frozen-checkpoint, "
                "no-retraining evaluations; run them with tools/tasks/evaluate.py, not the "
                "parallel launcher (which retrains via tools/launch.py)"
            )
        run_key = (family, experiment)
        if run_key in claimed_runs:
            raise SystemExit(f"duplicate parallel run: family={family}, experiment={experiment}")
        claimed_runs.add(run_key)
        command = (
            sys.executable,
            str(discover_code_root() / "tools" / "launch.py"),
            "--config",
            str(experiment_config),
            "--gpus",
            ",".join(map(str, group)),
            *job_arguments,
            *passthrough,
        )
        jobs.append(Job(experiment, experiment_config, group, wave, job_arguments, command))
    return jobs


def _preflight(jobs: list[Job], *, overwrite: bool) -> None:
    for job in jobs:
        overrides = [
            f"compute.devices={list(job.gpus)}",
            f"compute.strategy={infer_compute_strategy(job.gpus)}",
        ]
        config = load_config(job.config_path, overrides)
        stage = str(config["experiment"]["stage"])
        config["resume"] = "--resume" in job.arguments or stage == "silver"
        config["overwrite"] = overwrite
        require_preflight(config, ProjectPaths.resolve(config))


def _run_waves(jobs: list[Job]) -> int:
    for wave in sorted({job.wave for job in jobs}):
        selected = [job for job in jobs if job.wave == wave]
        processes = [(job, subprocess.Popen(job.command, text=True)) for job in selected]
        failed = False
        for job, process in processes:
            status = process.wait()
            print(f"{job.experiment} {'COMPLETED' if status == 0 else 'FAILED'}")
            failed |= status != 0
        if failed:
            print(f"Wave {wave + 1} failed; later waves were not started.")
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multiple independent experiments on reusable, disjoint GPU groups"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    parallel = _read_parallel_config(config_path)
    passthrough = ["--overwrite"] if args.overwrite else []
    jobs = _build_jobs(config_path, parallel, passthrough)
    print("=" * 72)
    print("PARALLEL EXPERIMENT PLAN")
    print("=" * 72)
    for job in jobs:
        print(
            f"wave={job.wave + 1} | {job.experiment} | "
            f"GPUs={','.join(map(str, job.gpus))} | {job.config_path}"
        )
    print("=" * 72)
    if args.dry_run:
        return 0
    _preflight(jobs, overwrite=args.overwrite)
    return _run_waves(jobs)


if __name__ == "__main__":
    raise SystemExit(main())
