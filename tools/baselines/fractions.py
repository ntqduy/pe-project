#!/usr/bin/env python
"""Training-fraction subsets of exp02: export their IDs and check them.

    python tools/baselines/fractions.py --dir <outputs>/<family>/BASE/<profile>/<task>/splits/data_fraction/seed_0

run_case.py calls ``export_subsets`` for every fraction case, so each seed's subsets are on
disk next to the runs that used them:

    <task dir>/splits/data_fraction/seed_<s>/frac_025.csv ... frac_100.csv   patient_id, study_id, label
    <task dir>/splits/data_fraction/seed_<s>/check.json                     result of check_subsets()

The subsets themselves come from source/data/experiment_splits.py (per-patient rank drawn with
the seed, stratified by label): only official TRAIN patients are subsampled, a patient keeps
all of its studies, validation and test never change. ``check_subsets`` verifies exactly that:
25 c 50 c 75 c 100 (patients and studies), no ID outside the official train split, no patient
split across subsets, and the positive rate of each subset close to the full train split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRACTIONS = (25, 50, 75, 100)
# |positive rate of a subset - positive rate of the full train split| above
# max(RATE_TOLERANCE, 2 / patients in the subset) is a FAIL: per-label ranks keep it within about
# one patient, which only becomes a small rate once the subset holds hundreds of patients.
RATE_TOLERANCE = 0.05


def _train_rows(manifest: Path, label: str) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return [{"patient_id": str(row["patient_id"]), "study_id": str(row["study_id"]), "label": row.get(label, "")}
                for row in csv.DictReader(handle) if row.get("split") == "train"]


def export_subsets(splits, source_manifest: str, label: str, destination: Path,
                   fractions: tuple[int, ...] = FRACTIONS) -> dict:
    """Write frac_<PPP>.csv for every fraction of one seed's assignment and check them."""
    destination.mkdir(parents=True, exist_ok=True)
    for percent in fractions:
        report = splits.materialize(source_manifest, percent / 100.0)
        rows = _train_rows(Path(report["manifest"]), label)
        with (destination / f"frac_{percent:03d}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient_id", "study_id", "label"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["patient_id"], row["study_id"])))
    official_train = _train_rows(splits.dataset_root / source_manifest, label)
    result = check_subsets(destination, official_train, fractions)
    result.update({"assignment": str(splits.root), "source_manifest": source_manifest, "label": label,
                   "split_seed": splits.seed})
    (destination / "check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _positive_rate(rows: list[dict]) -> float | None:
    values = []
    for row in rows:
        try:
            values.append(float(row["label"]))
        except (TypeError, ValueError):
            continue
    return sum(value > 0 for value in values) / len(values) if values else None


def check_subsets(directory: Path, official_train: list[dict] | None = None,
                  fractions: tuple[int, ...] = FRACTIONS) -> dict:
    """PASS / FAIL of the nesting, ID and class-balance rules for one seed's subsets."""
    subsets = {percent: _read(directory / f"frac_{percent:03d}.csv") for percent in fractions}
    problems: list[str] = []
    for smaller, larger in zip(fractions, fractions[1:]):
        for unit in ("patient_id", "study_id"):
            missing = {row[unit] for row in subsets[smaller]} - {row[unit] for row in subsets[larger]}
            if missing:
                problems.append(f"{len(missing)} {unit}s of {smaller}% are not in {larger}% (e.g. {sorted(missing)[:3]})")
    reference = official_train if official_train is not None else subsets[max(fractions)]
    train_studies = {row["study_id"] for row in reference}
    train_patients = {row["patient_id"] for row in reference}
    studies_of: dict[str, set[str]] = {}
    for row in reference:
        studies_of.setdefault(row["patient_id"], set()).add(row["study_id"])
    full_rate = _positive_rate(reference)
    rates = {}
    for percent, rows in subsets.items():
        outside = {row["study_id"] for row in rows} - train_studies
        if outside or {row["patient_id"] for row in rows} - train_patients:
            problems.append(f"{percent}%: {len(outside)} studies are not in the official train split")
        kept: dict[str, set[str]] = {}
        for row in rows:
            kept.setdefault(row["patient_id"], set()).add(row["study_id"])
        partial = [patient for patient, studies in kept.items() if studies != studies_of.get(patient, studies)]
        if partial:
            problems.append(f"{percent}%: {len(partial)} patients keep only some of their studies")
        rate = _positive_rate(rows)
        rates[f"{percent}%"] = {"patients": len(kept), "studies": len(rows), "positive_rate": rate}
        tolerance = max(RATE_TOLERANCE, 2.0 / max(1, len(kept)))
        if rate is not None and full_rate is not None and abs(rate - full_rate) > tolerance:
            problems.append(f"{percent}%: positive rate {rate:.3f} vs {full_rate:.3f} in the full train split "
                            f"(tolerance {tolerance:.3f})")
    if official_train is not None and {row["study_id"] for row in subsets[100]} != train_studies:
        problems.append("100% is not the whole official train split")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems,
            "train_positive_rate": full_rate, "subsets": rates}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help=".../splits/data_fraction/seed_<s>")
    args = parser.parse_args(argv)
    result = check_subsets(args.dir)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
