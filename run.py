#!/usr/bin/env python
"""Human entrypoint for the PE project.

This file contains no scientific logic. It resolves a semantic experiment name from
configs/experiments.yaml and delegates to the existing tools:

    python run.py list                          what can I run?
    python run.py show   <experiment>           what is this experiment, exactly?
    python run.py plan   <experiment>           what must exist first, and what is missing?
    python run.py preflight <experiment>        tools/preflight.py
    python run.py dry    <experiment> --gpus 0  tools/launch.py --dry-run
    python run.py run    <experiment> --gpus 0  tools/launch.py

`plan` never executes prerequisites; it only reports them. Data-generation stages (silver
generation, counterfactual inference) still require an explicit selection flag, exactly as
the underlying tools do: --patient-id, --max-cases/--max-reports, or --allow-full.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REGISTRY_PATH = ROOT / "configs" / "experiments.yaml"
GROUP_ORDER = (
    "DATA",
    "FOUNDATION",
    "REPRESENTATION",
    "PROBE",
    "DIAGNOSIS",
    "PROGNOSIS",
    "ANATOMY ANALYSIS",
    "DEFERRED",
)
# The two data-generation stages are not model training, so tools/launch.py deliberately
# does not know them: each has its own CLI with its own worker model (GPU study sharding for
# segmentation, CPU workers for ROI construction).
DATA_STAGE_TOOLS = {
    "segmentation": {"tool": "tools/create_masks/generate_masks.py", "gpus": True},
    "roi": {"tool": "tools/build_rois/build_rois.py", "gpus": False},
}
# Artifact paths are stored relative to different roots depending on the key.
OUTPUT_RELATIVE_KEYS = ("roi.segmentation_run", "init.checkpoint")
CODE_RELATIVE_KEYS = ("model.checkpoint", "model.repo", "segmentation.weights_directory")
DATASET_RELATIVE_KEYS = ("data.manifest", "silver.reports")


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_registry() -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        fail("PyYAML is required: pip install -r requirements.txt")
    if not REGISTRY_PATH.is_file():
        fail(f"experiment registry not found: {REGISTRY_PATH}")
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    experiments = payload.get("experiments") or {}
    if not isinstance(experiments, dict):
        fail("configs/experiments.yaml: 'experiments' must be a mapping")
    return {"experiments": experiments, "project_blockers": payload.get("project_blockers") or []}


def resolve_entry(registry: dict[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    experiments = registry["experiments"]
    if name not in experiments:
        matches = sorted(key for key in experiments if name in key)
        hint = ("\n  did you mean: " + ", ".join(matches)) if matches else ""
        fail(f"unknown experiment: {name}{hint}\n  run 'python run.py list' to see every name")
    return name, dict(experiments[name])


def config_path(entry: dict[str, Any]) -> Path:
    raw = str(entry.get("config") or "")
    if not raw:
        fail("registry entry has no config path")
    path = ROOT / raw
    if not path.is_file():
        fail(f"config file listed in the registry does not exist: {raw}")
    return path


def load_resolved_config(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a config through the project's own loader; None if it cannot be resolved."""
    try:
        from source.utils.config import load_config
    except ModuleNotFoundError as exc:
        print(f"note: cannot import the project config loader ({exc})")
        return None
    try:
        return load_config(config_path(entry))
    except Exception as exc:  # noqa: BLE001 - report, never crash the overview commands
        print(f"note: config did not resolve ({type(exc).__name__}: {exc})")
        return None


def dotted(config: dict[str, Any], key: str) -> Any:
    cursor: Any = config
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def artifact_path(config: dict[str, Any], key: str) -> Path | str | None:
    """Resolve a config value to a filesystem path using the project's path rules."""
    value = dotted(config, key)
    if value is None or not str(value).strip():
        return None
    text = str(value)
    if "${" in text:
        return "unresolved environment variable (set PE_CLOUD_ROOT)"
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    try:
        from source.data.paths import ProjectPaths

        paths = ProjectPaths.resolve(config)
        if key in OUTPUT_RELATIVE_KEYS:
            return paths.output_asset(candidate)
        if key in CODE_RELATIVE_KEYS:
            return paths.code_asset(candidate)
        if key in DATASET_RELATIVE_KEYS:
            return paths.dataset_root(str((config.get("data") or {}).get("mode") or "full")) / candidate
        return paths.code_asset(candidate)
    except Exception as exc:  # noqa: BLE001 - path roots may be unconfigured
        return f"unresolved ({type(exc).__name__})"


def requirement_state(
    requirement: dict[str, Any], config: dict[str, Any] | None, registry: dict[str, Any]
) -> tuple[str, str]:
    """Return (state, detail) for one requirement without touching any model or dataset."""
    if requirement.get("blocker"):
        return "BLOCKED", f"unresolved contract: {requirement['blocker']}"
    producer = requirement.get("experiment")
    if producer:
        produced_by = registry["experiments"].get(producer) or {}
        if str(produced_by.get("status")) == "unavailable":
            return "BLOCKED", f"{producer} is unavailable by design"
    key = requirement.get("path_key")
    if not key or config is None:
        return ("MISSING", f"produced by {producer}") if producer else ("READY", "no artifact to check")
    resolved = artifact_path(config, key)
    if resolved is None:
        return "BLOCKED", f"{key} is not configured"
    if isinstance(resolved, str):
        return "BLOCKED", resolved
    if resolved.exists():
        return "READY", str(resolved)
    return "MISSING", str(resolved)


def command_list(registry: dict[str, Any], _args: argparse.Namespace) -> int:
    experiments = registry["experiments"]
    print(f"{len(experiments)} experiments in configs/experiments.yaml\n")
    for group in GROUP_ORDER:
        members = {name: entry for name, entry in experiments.items() if entry.get("group") == group}
        if not members:
            continue
        print(group)
        width = max(len(name) for name in members)
        for name, entry in members.items():
            status = str(entry.get("status") or "?")
            alias = " (alias)" if entry.get("alias_of") else ""
            print(f"  {name:<{width}}  [{status}]{alias}  {entry.get('description', '')}")
        print()
    ungrouped = [name for name, entry in experiments.items() if entry.get("group") not in GROUP_ORDER]
    if ungrouped:
        print("UNGROUPED (fix the registry):", ", ".join(sorted(ungrouped)))
    print("Next: python run.py show <experiment>   |   python run.py plan <experiment>")
    return 0


def command_show(registry: dict[str, Any], args: argparse.Namespace) -> int:
    name, entry = resolve_entry(registry, args.experiment)
    config = load_resolved_config(entry)
    task = dict((config or {}).get("task") or {})
    lineage = dict((config or {}).get("lineage") or {})
    experiment = dict((config or {}).get("experiment") or {})

    print(f"\n{name}   [{entry.get('status', '?')}]   group: {entry.get('group', '?')}")
    if entry.get("alias_of"):
        print(f"alias of        : {entry['alias_of']} (same config, same run directory)")
    print(f"description     : {entry.get('description', '-')}")
    print(f"question        : {entry.get('question', '-')}")
    print(f"config          : {entry.get('config')}")
    print(f"internal id     : {experiment.get('id', '?')}  (stage: {experiment.get('stage', '?')})")
    if entry.get("legacy_id"):
        print(f"legacy id       : {entry['legacy_id']} (carried over unchanged)")
    print(f"representation  : {entry.get('representation', '-')}")
    if lineage.get("source_checkpoint"):
        print(f"  checkpoint    : {lineage['source_checkpoint']}")
    print(f"input modalities: {', '.join(entry.get('modalities') or []) or '-'}")
    print(f"anatomy branches: {entry.get('anatomy', '-')}"
          f"   (task.regions: {task.get('regions', '-')})")
    print(f"supervision     : {entry.get('supervision', '-')}")
    print(f"fusion          : {entry.get('fusion', '-')}")
    print(f"peft            : {(config or {}).get('peft', {}).get('method', '-')}")

    print("\nrequires:")
    for requirement in entry.get("requires") or []:
        state, detail = requirement_state(dict(requirement), config, registry)
        producer = f" (from {requirement['experiment']})" if requirement.get("experiment") else ""
        print(f"  [{state:<7}] {requirement.get('what')}{producer}")
        print(f"            {detail}")
    if not entry.get("requires"):
        print("  (nothing)")

    print("\nproduces:")
    for item in entry.get("produces") or ["(nothing)"]:
        print(f"  - {item}")

    blockers = list(entry.get("blockers") or [])
    print("\nblockers:")
    if blockers:
        for blocker in blockers:
            print(f"  ! {blocker}")
    else:
        print("  none recorded for this experiment")
    if config is not None and (config.get("model") or {}).get("backbone"):
        for blocker in registry["project_blockers"]:
            print(f"  ! [project:{blocker.get('id')}] {str(blocker.get('what', '')).strip()}")
    print("\ncheck it yourself: python run.py preflight " + name)
    return 0


def command_plan(registry: dict[str, Any], args: argparse.Namespace) -> int:
    name, _ = resolve_entry(registry, args.experiment)
    print(f"\ndependency chain for {name} (nothing is executed)\n")
    worst: list[str] = []
    expanded: set[str] = set()

    def walk(current: str, depth: int, seen: tuple[str, ...]) -> None:
        entry = dict(registry["experiments"].get(current) or {})
        indent = "  " * depth
        if current in seen:
            print(f"{indent}{current}  [cycle - already listed]")
            return
        if current in expanded:
            print(f"{indent}{current}  [{entry.get('status', '?')}] (chain already shown above)")
            return
        expanded.add(current)
        config = load_resolved_config(entry) if entry.get("config") else None
        status = str(entry.get("status") or "?")
        print(f"{indent}{current}  [{status}]")
        for requirement in entry.get("requires") or []:
            requirement = dict(requirement)
            state, detail = requirement_state(requirement, config, registry)
            worst.append(state)
            print(f"{indent}  <- [{state:<7}] {requirement.get('what')}")
            print(f"{indent}     {detail}")
            producer = requirement.get("experiment")
            if producer and producer in registry["experiments"] and producer != current:
                walk(producer, depth + 2, (*seen, current))

    walk(name, 0, ())
    summary = "READY" if all(state == "READY" for state in worst) else (
        "BLOCKED" if "BLOCKED" in worst else "MISSING")
    print(f"\noverall: {summary}")
    print("  READY   the artifact exists now")
    print("  MISSING the artifact does not exist yet; run the experiment that produces it")
    print("  BLOCKED an unresolved data/checkpoint contract must be filled first"
          " (see docs/BLOCKERS.md)")
    if summary != "READY":
        print("\nNothing was run. Prerequisites are never executed implicitly.")
    return 0


def experiment_stage(entry: dict[str, Any]) -> str:
    config = load_resolved_config(entry)
    return str(((config or {}).get("experiment") or {}).get("stage") or "")


def delegate(entry: dict[str, Any], name: str, args: argparse.Namespace, *, mode: str) -> int:
    """Run one experiment through the CLI that owns its stage.

    mode is 'preflight', 'dry' or 'run'. Training and generation stages go through
    tools/launch.py; the two data stages have their own CLIs (DATA_STAGE_TOOLS), which take a
    different flag set — passing launch.py flags to them would just fail with argparse noise.
    """
    status = str(entry.get("status") or "")
    if status == "unavailable" and mode != "preflight":
        print(f"refusing to {mode} {name}: it is unavailable by design.")
        for blocker in entry.get("blockers") or []:
            print(f"  ! {blocker}")
        print(f"Run 'python run.py show {name}' for the contract it would need.")
        return 3
    if entry.get("alias_of"):
        print(f"note: {name} is an alias of {entry['alias_of']} and shares its run directory.")
    if status == "deferred" and mode != "preflight":
        print(f"note: {name} is deferred from the active pipeline; continuing anyway.")

    stage = experiment_stage(entry)
    data_stage = DATA_STAGE_TOOLS.get(stage)
    selected = bool(args.patient_id or args.allow_full
                    or args.max_cases is not None or args.max_reports is not None)
    if mode == "preflight":
        tool = ROOT / "tools" / "preflight.py"
        accepts = {"gpus": True, "set": True, "state": True, "selection": False}
    elif data_stage:
        if mode == "dry":
            gpus = " --gpus 0" if data_stage["gpus"] else ""
            print(f"{name} runs stage '{stage}', whose CLI has no --dry-run.")
            print(f"Check it instead with: python run.py preflight {name}{gpus}")
            return 3
        if not selected:
            print(f"stage '{stage}' needs an explicit scope so a full run is never accidental:")
            print(f"  python run.py run {name} --patient-id PATIENT_001")
            print(f"  python run.py run {name} --max-cases 5")
            print(f"  python run.py run {name} --allow-full")
            return 3
        tool = ROOT / data_stage["tool"]
        accepts = {"gpus": data_stage["gpus"], "set": False, "state": False, "selection": True}
    else:
        tool = ROOT / "tools" / "launch.py"
        accepts = {"gpus": True, "set": True, "state": True, "selection": True}

    command = [sys.executable, str(tool), "--config", str(config_path(entry))]
    if args.gpus is not None:
        if accepts["gpus"]:
            command += ["--gpus", args.gpus]
        else:
            print(f"note: stage '{stage}' parallelizes over CPU workers (roi.workers);"
                  " ignoring --gpus.")
    if args.set and not accepts["set"]:
        fail(f"stage '{stage}' has no --set support; edit {entry.get('config')} instead")
    for override in args.set or []:
        command += ["--set", override]
    if args.resume:
        if not accepts["state"]:
            fail(f"stage '{stage}' continues a partial run by default; --resume is not accepted"
                 " (use --overwrite to discard the previous run)")
        command.append("--resume")
    if args.overwrite:
        command.append("--overwrite")
    if selected and not accepts["selection"]:
        print("note: preflight validates the whole configured input; ignoring scope flags.")
    elif accepts["selection"]:
        for patient in args.patient_id or []:
            command += ["--patient-id", patient]
        if args.max_cases is not None:
            command += ["--max-cases", str(args.max_cases)]
        if args.max_reports is not None:
            command += ["--max-reports", str(args.max_reports)]
        if args.allow_full:
            command.append("--allow-full")
    if mode == "dry":
        command.append("--dry-run")
    print("delegating to:", " ".join(command))
    return subprocess.call(command, cwd=str(ROOT))


def command_preflight(registry: dict[str, Any], args: argparse.Namespace) -> int:
    name, entry = resolve_entry(registry, args.experiment)
    return delegate(entry, name, args, mode="preflight")


def command_dry(registry: dict[str, Any], args: argparse.Namespace) -> int:
    name, entry = resolve_entry(registry, args.experiment)
    return delegate(entry, name, args, mode="dry")


def command_run(registry: dict[str, Any], args: argparse.Namespace) -> int:
    name, entry = resolve_entry(registry, args.experiment)
    return delegate(entry, name, args, mode="run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="List, inspect, plan, check and launch PE experiments by semantic name.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="print every experiment, grouped by pipeline stage")

    def with_experiment(name: str, help_text: str, passthrough: bool) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("experiment")
        if passthrough:
            sub.add_argument("--gpus", default=None, help="physical GPU IDs, for example 0,1")
            sub.add_argument("--set", action="append", metavar="KEY=VALUE",
                             help="config override, repeatable")
            state = sub.add_mutually_exclusive_group()
            state.add_argument("--resume", action="store_true")
            state.add_argument("--overwrite", action="store_true")
            selection = sub.add_mutually_exclusive_group()
            selection.add_argument("--patient-id", action="append", metavar="ID",
                                   help="one real patient, repeatable")
            selection.add_argument("--max-cases", type=int)
            selection.add_argument("--max-reports", type=int)
            selection.add_argument("--allow-full", action="store_true")
        return sub

    with_experiment("show", "print what an experiment is and what it needs", False)
    with_experiment("plan", "print the dependency chain, marked READY/MISSING/BLOCKED", False)
    with_experiment("preflight", "delegate to tools/preflight.py", True)
    with_experiment("dry", "delegate to tools/launch.py --dry-run", True)
    with_experiment("run", "delegate to tools/launch.py", True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not os.environ.get("PE_CLOUD_ROOT"):
        print("note: PE_CLOUD_ROOT is not set, so data and output paths cannot be checked.")
    registry = load_registry()
    handlers = {
        "list": command_list,
        "show": command_show,
        "plan": command_plan,
        "preflight": command_preflight,
        "dry": command_dry,
        "run": command_run,
    }
    return handlers[args.command](registry, args)


if __name__ == "__main__":
    raise SystemExit(main())
