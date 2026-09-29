#!/usr/bin/env python
"""Report how far a task run already got, for the same --config/--set as tools/launch.py.

Prints one line, ``<state> <run_dir> [<sections>]``, where run_dir is the folder train_task.py
and evaluate.py use (``<outputs>/<family>/<id>/epoch_<training.epochs>``) and state is:

    absent      no run folder yet
    incomplete  the folder exists but training never recorded a completed result
    trained     result.json says completed and checkpoint/best.ckpt exists
    evaluated   trained, and the full-test evaluation wrote result.csv, predictions.csv and
                its section of result.json
    different   a completed run whose resolved_config.yaml differs from the requested config;
                <sections> lists the differing top-level sections (e.g. training)

The folder name only carries the id and the epoch budget, so a completed run made with, say,
another early-stopping patience sits in the same folder; it reads as different, not as done.
The run scripts use this to skip stages whose output already exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.data.paths import ProjectPaths
from source.engine.experiment import OutputManager, run_output_id

from tools._common import base_parser, resolve_cli_config

# Written alongside every run but not part of what it computes: bookkeeping, where things are
# stored (the dataset itself is data.profile + data.manifest), and which GPUs and how many
# DataLoader workers happened to be used (precision and strategy still count).
IGNORED_SECTIONS = {"paths", "resolved_paths", "config_hash", "resume", "overwrite"}
IGNORED_COMPUTE_KEYS = {"devices", "num_workers"}


def _comparable(config: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in config.items() if key not in IGNORED_SECTIONS}
    compute = result.get("compute")
    if isinstance(compute, dict):
        result["compute"] = {key: value for key, value in compute.items() if key not in IGNORED_COMPUTE_KEYS}
    return result


def settings_difference(run_dir: Path, config: dict[str, Any]) -> list[str]:
    """Top-level config sections in which the stored run differs from ``config``."""
    try:
        stored = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return ["resolved_config.yaml"]
    if not isinstance(stored, dict):
        return ["resolved_config.yaml"]
    old, new = _comparable(stored), _comparable(config)
    return sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))


def run_state(run_dir: Path) -> str:
    if not run_dir.exists():
        return "absent"
    try:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "incomplete"
    if (result.get("experiment") or {}).get("status") != "completed":
        return "incomplete"
    if not (run_dir / "checkpoint" / "best.ckpt").is_file():
        return "incomplete"
    # evaluate.py writes result.csv and predictions.csv first and result.json last, so a
    # run interrupted mid-evaluation still reads as trained.
    evaluated = (
        (result.get("evaluation_scope") or {}).get("mode") == "full_test"
        and (run_dir / "result.csv").is_file()
        and (run_dir / "predictions.csv").is_file()
    )
    return "evaluated" if evaluated else "trained"


def main() -> int:
    parser = base_parser("Report whether a task run is absent, incomplete, trained or evaluated")
    args = parser.parse_args()
    config = resolve_cli_config(args)
    paths = ProjectPaths.resolve(config)
    family = str(config["experiment"].get("family") or config["experiment"]["stage"])
    run_dir = OutputManager(paths).run_dir(family, run_output_id(config))
    state = run_state(run_dir)
    if state in {"trained", "evaluated"}:
        difference = settings_difference(run_dir, config)
        if difference:
            print(f"different {run_dir} {','.join(difference)}")
            return 0
    print(f"{state} {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
