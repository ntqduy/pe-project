#!/usr/bin/env python
"""Write the tiny synthetic dataset behind ``run_case.py --smoke`` (no patient data).

    python tools/baselines/smoke_data.py                    # -> $PE_SMOKE_ROOT or output/_smoke
    python tools/baselines/smoke_data.py --root /tmp/pe_smoke --force

Layout (the ``smoke_30`` profile of a project whose ``paths.*`` all point under ``<root>``):

    <root>/raw/                                      empty stand-in for the raw INSPECT mount
    <root>/derived/datasets/smoke_30/dataset.json
        manifests/diagnosis.csv                      image_path, pe_present (0/1)
        manifests/prognosis_{all_patient,pe_positive}.csv   + the 7 outcomes (1 / 0 / empty)
        manifests/ct_fm/<same three names>           pooled_path instead of image_path
        volumes/<study>.npy                          float32 [1, S, S, S] in [0, 1]
        ct_fm/pooled/<study>.npy (+ .metadata.json)  float32 [513, 1, 1, 1], channel 512 = 1
    <root>/dummy_ct_fm.safetensors                   stand-in checkpoint path for ct_fm_features
    <root>/pe-project/outputs/                       where smoke runs write

19 patients / 25 studies split by patient (train 6 / validation 5 / test 8 patients); the first
two patients of every split have two studies. Every split holds both classes of pe_present and
of every prognosis outcome (also inside the PE-positive cohort). Positive studies carry a
brighter blob, so a model can learn something in two epochs; nothing here is meant to be
scored, only to drive the pipeline end to end.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.baselines.experiments import PROGNOSIS_LABELS  # noqa: E402

PROFILE = "smoke_30"
# registry.ct_fm_features weight_sha256 (configs/components/backbones.yaml): the pooled-feature
# sidecars must name the same CT-FM weights the config expects.
CT_FM_WEIGHT_SHA256 = "b521ff13764ad0fce67f8ad2e5aa9ccc0823f69e4544b446581d8a2dee686215"
SPLIT_PATIENTS = {"train": 6, "validation": 5, "test": 8}
MARKER = "smoke_dataset.json"


def default_root() -> Path:
    return Path(os.environ.get("PE_SMOKE_ROOT") or ROOT / "output" / "_smoke")


def path_overrides(root: Path) -> list[str]:
    """--set values that point every project path of a run at the smoke root."""
    base = root.resolve().as_posix()
    return [
        f"paths.cloud_root={base}",
        f"paths.cloud_project_root={base}/pe-project",
        f"paths.output_root={base}/pe-project/outputs",
        f"paths.data_root={base}/data",
        f"paths.raw_inspect={base}/raw",
        f"paths.derived_data={base}/derived",
    ]


def dataset_root(root: Path) -> Path:
    return root / "derived" / "datasets" / PROFILE


def _studies() -> list[dict]:
    rows = []
    patient_number, study_number = 900001, 1
    for split, patients in SPLIT_PATIENTS.items():
        position = 0
        for patient in range(patients):
            for _ in range(2 if patient < 2 else 1):
                rows.append({"patient_id": str(patient_number), "study_id": f"SMK{study_number:04d}",
                             "split": split, "position": position})
                study_number += 1
                position += 1
            patient_number += 1
    for row in rows:
        row["pe_present"] = row.pop("position") % 2
    # Outcomes alternate inside every (split, pe_present) group, so each split holds both
    # classes in the all-comers and in the PE-positive cohort; in groups of 4+ rows the third
    # row is censored (empty) for every other outcome.
    for split in SPLIT_PATIENTS:
        for pe in (0, 1):
            group = [row for row in rows if row["split"] == split and row["pe_present"] == pe]
            for j, row in enumerate(group):
                for k, label in enumerate(PROGNOSIS_LABELS):
                    censored = len(group) >= 4 and j == 2 and k % 2 == 0
                    row[label] = "" if censored else (j + k) % 2
    return rows


def _check_classes(rows: list[dict], name: str, labels: tuple[str, ...]) -> None:
    for split in SPLIT_PATIENTS:
        for label in labels:
            values = {row[label] for row in rows if row["split"] == split and row[label] != ""}
            if values != {0, 1}:
                raise AssertionError(f"{name}: {split} lacks a class of {label} ({values})")


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_dataset(root: Path, *, size: int = 64, seed: int = 0, force: bool = False) -> Path:
    import numpy as np

    data = dataset_root(root)
    marker = data / MARKER
    spec = {"size": size, "seed": seed, "patients": sum(SPLIT_PATIENTS.values()), "version": 1}
    if marker.is_file() and not force and json.loads(marker.read_text(encoding="utf-8")) == spec:
        return data
    if data.exists():
        shutil.rmtree(data)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "pe-project" / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "dummy_ct_fm.safetensors").write_bytes(b"synthetic smoke stand-in, never loaded\n")
    rows = _studies()
    generator = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*(np.arange(size),) * 3, indexing="ij"))
    for row in rows:
        volume = np.clip(generator.normal(0.3, 0.08, (size, size, size)), 0.0, 1.0)
        if row["pe_present"]:
            centre = generator.integers(size // 4, 3 * size // 4, 3)
            blob = ((grid - centre[:, None, None, None]) ** 2).sum(0) <= (size // 8) ** 2
            volume[blob] = np.clip(volume[blob] + 0.5, 0.0, 1.0)
        relative = f"volumes/{row['study_id']}.npy"
        (data / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(data / relative, volume[None].astype(np.float32))
        row["image_path"] = relative
        pooled = generator.normal(0.0, 1.0, 513).astype(np.float32)
        pooled[:8] += 1.5 * row["pe_present"]
        pooled[512] = 1.0
        pooled_relative = f"ct_fm/pooled/{row['study_id']}.npy"
        (data / pooled_relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(data / pooled_relative, pooled.reshape(513, 1, 1, 1))
        (data / f"{pooled_relative}.metadata.json").write_text(json.dumps(
            {"representation": "ct_fm_features_v1", "weight_sha256": CT_FM_WEIGHT_SHA256}), encoding="utf-8")
        row["pooled_path"] = pooled_relative
    positive = [row for row in rows if row["pe_present"] == 1]
    _check_classes(rows, "diagnosis", ("pe_present",))
    _check_classes(rows, "prognosis_all_patient", PROGNOSIS_LABELS)
    _check_classes(positive, "prognosis_pe_positive", PROGNOSIS_LABELS)
    ids = ["patient_id", "study_id", "split"]
    for folder, file_column in (("manifests", "image_path"), ("manifests/ct_fm", "pooled_path")):
        _write_csv(data / folder / "diagnosis.csv", rows, [*ids, file_column, "pe_present"])
        _write_csv(data / folder / "prognosis_all_patient.csv", rows, [*ids, file_column, "pe_present", *PROGNOSIS_LABELS])
        _write_csv(data / folder / "prognosis_pe_positive.csv", positive, [*ids, file_column, "pe_present", *PROGNOSIS_LABELS])
    (data / "dataset.json").write_text(json.dumps({
        "profile": PROFILE, "synthetic": True,
        "cohort": {"patients": spec["patients"], "studies": len(rows)},
        "preprocessing": {"fingerprint": f"synthetic-smoke-{size}-{seed}", "shape": [size] * 3},
    }, indent=2), encoding="utf-8")
    marker.write_text(json.dumps(spec), encoding="utf-8")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--size", type=int, default=64, help="cube edge; 64 keeps ResNet-3D layer4 at 2^3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="rewrite even if an identical dataset exists")
    args = parser.parse_args(argv)
    data = write_dataset(args.root, size=args.size, seed=args.seed, force=args.force)
    print(f"synthetic smoke dataset: {data}")
    print("run settings: " + " ".join(f"--set {item}" for item in path_overrides(args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
