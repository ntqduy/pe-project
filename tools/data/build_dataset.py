#!/usr/bin/env python
"""Build one dataset profile from the read-only INSPECT release.

This tool contains no scientific logic. It resolves the run config, loads the dataset
profile it names, and hands both to ``source/data_preprocessing.pipeline.build_dataset``,
which is the single implementation every profile shares.

    python run.py run data.dataset.test_500_sample --max-cases 10     smoke
    python run.py run data.dataset.test_500_sample --allow-full       whole profile
    python run.py run data.dataset.full_inspect    --allow-full       whole cohort

Like every other generation stage, a full build needs an explicit scope flag, so it can
never happen by accident.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.data_preprocessing.pipeline import build_dataset
from source.dataset import require_active_profile
from source.utils.config import load_config, validate_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--max-cases", type=int, help="build only the first N patients (smoke test)")
    scope.add_argument("--allow-full", action="store_true", help="build the whole profile")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="write manifests that point at the raw read-only volumes instead of a cache",
    )
    parser.add_argument("--metadata-only", action="store_true",
                        help="skip file-existence rules; validate the cohort from metadata alone")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    config = validate_config({key: value for key, value in config.items() if key != "config_hash"})
    if args.max_cases is None and not args.allow_full:
        raise SystemExit(
            "building a dataset needs an explicit scope so a full build is never accidental:\n"
            "  --max-cases 10   smoke test\n"
            "  --allow-full     the whole profile"
        )
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")

    name = str(config["data"]["profile"])
    profile = require_active_profile(name)
    paths = ProjectPaths.resolve(config)
    # Data stages bypass tools/launch.py, so enforce the same dependency/source/output
    # gate here before creating a destination or extracting the large EHR archive.
    require_preflight(config, paths)
    dataset_config = dict(config.get("dataset") or {})
    payload = build_dataset(
        profile,
        paths,
        max_cases=args.max_cases,
        allow_full=bool(args.allow_full),
        preprocess=not args.no_preprocess and bool(dataset_config.get("preprocess", True)),
        overwrite=bool(args.overwrite),
        check_files=False if args.metadata_only else None,
    )
    print(json.dumps(payload["cohort"] | {"output": payload["output_root"]}, indent=2))
    print(f"exclusions: {payload['eligibility']['excluded_by_rule']}")
    print(f"split patients: {payload['split_audit']['split_patients']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
