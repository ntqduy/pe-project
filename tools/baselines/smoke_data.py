#!/usr/bin/env python
"""Write the tiny synthetic dataset behind ``run_case.py --smoke`` (no patient data).

    python tools/baselines/smoke_data.py                    # -> $PE_SMOKE_ROOT or output/_smoke
    python tools/baselines/smoke_data.py --root /tmp/pe_smoke --force

Layout (the ``smoke_30`` profile of a project whose ``paths.*`` all point under ``<root>``):

    <root>/raw/<study>.nii.gz                        synthetic CT in HU (body, lungs, PE-positive blob)
    <root>/derived/datasets/smoke_30/dataset.json
        manifests/diagnosis.csv                      image_path, pe_present (0/1)
        manifests/prognosis_{all_patient,pe_positive}.csv   + the 7 outcomes (1 / 0 / empty)
        manifests/ct_fm/<same three names>           pooled_path instead of image_path
        volumes/<study>.npy (+ .metadata.json)       the project's own cache preprocessing of raw/
                                                     (source/data/build/volumes.py, profile spec at S^3)
        ct_fm/pooled/<study>.npy (+ .metadata.json)  float32 [513, 1, 1, 1], channel 512 = 1, with the
                                                     tools/data/build_ctfm_cache.py sidecar fields
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


def _cache_spec(size: int, orientation: str):
    """The profile's preprocessing contract (source/data/profiles/_common.yaml) at size^3."""
    import yaml

    from source.data.build.volumes import PreprocessingSpec

    profile = yaml.safe_load((ROOT / "source" / "data" / "profiles" / "_common.yaml").read_text(encoding="utf-8"))
    payload = {**dict(profile["preprocessing"]), "target_shape": [size] * 3, "orientation": orientation,
               "emit_patch_grid": False}
    return PreprocessingSpec.from_mapping(payload)


def _raw_ct(generator, positive: bool):
    """A synthetic chest CT in HU: air, body, two lungs and, for PE-positive studies, a bright blob."""
    import nibabel as nib
    import numpy as np

    shape = (80, 80, 48)
    x, y, z = np.meshgrid(*(np.linspace(-1, 1, n) for n in shape), indexing="ij")
    volume = np.full(shape, -1000.0, dtype=np.float32)
    volume[(x / 0.85) ** 2 + (y / 0.65) ** 2 <= 1] = 40.0
    for side in (-0.4, 0.4):
        volume[((x - side) / 0.3) ** 2 + (y / 0.45) ** 2 <= 1] = -820.0
    volume += generator.normal(0.0, 25.0, shape).astype(np.float32)
    if positive:
        centre = generator.uniform(-0.3, 0.3, 3)
        volume[((x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2) <= 0.12 ** 2] = 250.0
    return nib.Nifti1Image(volume, np.diag([-0.9, -0.9, 2.0, 1.0]))


def write_dataset(root: Path, *, size: int = 64, seed: int = 0, force: bool = False) -> Path:
    import nibabel as nib
    import numpy as np

    from source.data.build.volumes import preprocess_study, preprocess_volume_with_metadata
    from tools.data.build_ctfm_cache import _feature_grid_affine

    data = dataset_root(root)
    marker = data / MARKER
    spec = {"size": size, "seed": seed, "patients": sum(SPLIT_PATIENTS.values()), "version": 2}
    if marker.is_file() and not force and json.loads(marker.read_text(encoding="utf-8")) == spec:
        return data
    if data.exists():
        shutil.rmtree(data)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "pe-project" / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "dummy_ct_fm.safetensors").write_bytes(b"synthetic smoke stand-in, never loaded\n")
    rows = _studies()
    generator = np.random.default_rng(seed)
    dense_spec, ctfm_spec = _cache_spec(size, "RAS"), _cache_spec(size, "SPL")
    for row in rows:
        study = row["study_id"]
        raw_path = root / "raw" / f"{study}.nii.gz"
        nib.save(_raw_ct(generator, bool(row["pe_present"])), str(raw_path))
        # Dense model input: exactly the project's cache preprocessing, sidecar included.
        written = preprocess_study(study, raw_path, data / "volumes", dense_spec, overwrite=True)
        row["image_path"] = Path(written["preprocessed_path"]).relative_to(data).as_posix()
        # Frozen CT-FM input: pooled features (synthetic values) with the sidecar fields of
        # tools/data/build_ctfm_cache.py, so the Grad-CAM preview can rebuild its CT canvas.
        _, metadata = preprocess_volume_with_metadata(raw_path, ctfm_spec)
        pooled = generator.normal(0.0, 1.0, 513).astype(np.float32)
        pooled[:8] += 1.5 * row["pe_present"]
        pooled[512] = 1.0
        pooled_relative = f"ct_fm/pooled/{study}.npy"
        (data / pooled_relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(data / pooled_relative, pooled.reshape(513, 1, 1, 1))
        cell = (size, size, size)
        (data / f"{pooled_relative}.metadata.json").write_text(json.dumps({
            **metadata, "status": "completed", "representation": "ct_fm_features_v1",
            "weight_sha256": CT_FM_WEIGHT_SHA256, "study_id": study, "raw_image_path": str(raw_path),
            "spec": ctfm_spec.as_dict(), "preprocessing_fingerprint": ctfm_spec.fingerprint(),
            "tensor_shape": [513, 1, 1, 1], "feature_channels": 512, "synthetic": True,
            "feature_grid": {"shape": [1, 1, 1], "cell_voxels": list(cell),
                             "affine": _feature_grid_affine(np.asarray(metadata["output_affine"]), cell).tolist()},
        }, default=str), encoding="utf-8")
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
        "preprocessing": {**dense_spec.as_dict(), "fingerprint": dense_spec.fingerprint()},
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
