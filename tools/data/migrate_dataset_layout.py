#!/usr/bin/env python3
"""Move one existing profile to the compact dataset / external-cache layout.

Dry-run by default. This never deletes an artifact or overwrites an existing target.
Run only after the derived storage is mounted and no preprocessing/training is active.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _replace_paths(value: object, replacements: list[tuple[str, str]]) -> object:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    return value


def _rewrite_csv(path: Path, replacements: list[tuple[str, str]]) -> bool:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            return False
        rows = list(reader)
    changed = False
    for row in rows:
        for field, value in row.items():
            if isinstance(value, str):
                updated = _replace_paths(value, replacements)
                if updated != value:
                    row[field] = updated
                    changed = True
    if changed:
        temporary = path.with_name(f".{path.name}.migration.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="actually move and rewrite artifacts")
    args = parser.parse_args()

    dataset = args.dataset_root.resolve()
    if dataset.parent.name != "datasets" or not dataset.name or dataset.name in {".", ".."}:
        parser.error("expected <derived>/datasets/<profile>")
    if not (dataset / "manifests").is_dir():
        parser.error("manifests/ is required (also for an incomplete build)")
    cache = dataset.parent.parent / "cache" / dataset.name
    moves: list[tuple[Path, Path]] = []
    for name in ("volumes", "clinical", "ct_fm_frozen"):
        source = dataset / name
        if source.exists():
            moves.append((source, cache / name))
    exclusions = dataset / "audit" / "exclusions.csv"
    if exclusions.is_file():
        moves.append((exclusions, dataset / "manifests" / "exclusions.csv"))
    audit = dataset / "audit"
    if audit.is_dir():
        moves.append((audit, cache / "legacy_audit"))
    redundant_quality = dataset / "data_quality.json"
    if redundant_quality.is_file():
        moves.append((redundant_quality, cache / "legacy_reports" / "data_quality.json"))
    ctfm_manifests = dataset / "ct_fm_frozen" / "manifests"
    if ctfm_manifests.is_dir():
        moves.append((ctfm_manifests, dataset / "manifests" / "ct_fm_frozen"))

    # Check all targets before making any changes; never silently merge or overwrite.
    for source, target in moves:
        if target.exists():
            parser.error(f"destination already exists: {target}")
    if not moves:
        print(f"already compact (or no legacy directories): {dataset}")
        return 0
    print(f"dataset: {dataset}")
    for source, target in moves:
        print(f"{source} -> {target}")
    if not args.apply:
        print("dry run only; add --apply after reviewing the moves")
        return 0

    # Move the nested CT-FM manifests first, before moving their parent cache.
    ordered = sorted(moves, key=lambda pair: -len(pair[0].parts))
    for source, target in ordered:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))

    replacements = [
        (str(dataset / "ct_fm_frozen" / "manifests"), str(dataset / "manifests" / "ct_fm_frozen")),
        ("ct_fm_frozen/manifests/", "manifests/ct_fm_frozen/"),
        (str(dataset / "ct_fm_frozen"), str(cache / "ct_fm_frozen")),
        ("ct_fm_frozen/preprocessing_failures.csv", str(cache / "ct_fm_frozen" / "preprocessing_failures.csv")),
        ("ct_fm_frozen/dropped_rows.csv", str(cache / "ct_fm_frozen" / "dropped_rows.csv")),
        (str(dataset / "clinical"), str(cache / "clinical")),
        (str(dataset / "volumes"), str(cache / "volumes")),
        ("audit/exclusions.csv", "manifests/exclusions.csv"),
    ]
    rewritten: list[str] = []
    for path in (dataset / "manifests").rglob("*.csv"):
        if _rewrite_csv(path, replacements):
            rewritten.append(str(path.relative_to(dataset)))
    metadata = dataset / "dataset.json"
    if metadata.is_file():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload = _replace_paths(payload, replacements)
        temporary = metadata.with_name(".dataset.json.migration.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        temporary.replace(metadata)
    quality = dataset / "data_quality.md"
    if quality.is_file():
        original = quality.read_text(encoding="utf-8")
        updated = _replace_paths(original, replacements)
        if updated != original:
            temporary = quality.with_name(".data_quality.md.migration.tmp")
            temporary.write_text(updated, encoding="utf-8")
            temporary.replace(quality)
    log = dataset / "logs.txt"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} | migrated dataset cache to {cache}; "
            f"rewrote {len(rewritten)} manifests; prior terminal output cannot be reconstructed\n"
        )
    print(f"migrated; rewritten manifests: {', '.join(rewritten) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
