#!/usr/bin/env python
"""Write k-fold / training-fraction manifests for the baseline experiments.

    python tools/data/build_split_manifests.py --profile full_inspect \
        --base-manifest manifests/diagnosis.csv --label pe_present --folds 5 --seed 42 \
        --apply manifests/diagnosis.csv manifests/ct_fm/diagnosis.csv \
        --fold official 0 1 2 3 4 --fraction 25 50 75 100

The official test split is never modified; see source/data/experiment_splits.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.experiment_splits import ExperimentSplits, fraction_tag, parse_fraction


def dataset_root(profile: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    import yaml

    from source.data.paths import ProjectPaths
    from source.utils.config import expand_environment

    text = (Path(__file__).resolve().parents[2] / "configs" / "paths.yaml").read_text(encoding="utf-8")
    paths = ProjectPaths.resolve(expand_environment(yaml.safe_load(text)))
    return paths.dataset_root("full", profile)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="full_inspect")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--task", default="diagnosis")
    parser.add_argument("--base-manifest", default="manifests/diagnosis.csv")
    parser.add_argument("--label", default="pe_present")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", nargs="+", default=None, help="manifests to derive (default: the base)")
    parser.add_argument("--fold", nargs="+", default=["official"])
    parser.add_argument("--fraction", nargs="+", default=["100"],
                        help="training percent (25, 12.5, 1%%) or fraction below 1 (0.25); a bare 1 is rejected")
    args = parser.parse_args()
    # Every value up front: a bad later --fraction must not stop the run after earlier
    # manifests were already written.
    fractions: list[float] = []
    for value in args.fraction:
        try:
            fraction = parse_fraction(value)
            fraction_tag(fraction)
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid --fraction {value!r}: {exc}")
        fractions.append(fraction)
    root = dataset_root(args.profile, args.dataset_root)
    splits = ExperimentSplits(root, task=args.task, label=args.label, base_manifest=args.base_manifest,
                              folds=args.folds, seed=args.seed)
    for manifest in args.apply or [args.base_manifest]:
        for fold in args.fold:
            for fraction in fractions:
                report = splits.materialize(manifest, fold, fraction)
                report["manifest"] = str(Path(report["manifest"]).relative_to(root))
                print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
