from __future__ import annotations

import csv
import json

import pytest


np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
pytest.importorskip("scipy")

from source.data_preprocessing.volumes import (
    PREPROCESSING_IMPLEMENTATION,
    PreprocessingSpec,
    preprocess_study,
    validate_cache_entry,
)


def _synthetic_ct(path):
    """A LPS-oriented anisotropic scan with body surrounded by scanner air."""
    volume = np.full((20, 18, 12), -1000.0, dtype=np.float32)
    volume[3:18, 2:16, 1:11] = -600.0
    volume[8:12, 7:11, 4:8] = 250.0
    affine = np.array(
        [[-2.0, 0.0, 0.0, 40.0], [0.0, -2.0, 0.0, 36.0], [0.0, 0.0, 3.0, -12.0], [0.0, 0.0, 0.0, 1.0]]
    )
    nib.save(nib.Nifti1Image(volume, affine), path)


def _spec() -> PreprocessingSpec:
    return PreprocessingSpec(
        target_shape=(16, 16, 16),
        orientation="RAS",
        resample_spacing_mm=(1.0, 1.0, 1.0),
        foreground_strategy="external_body",
        foreground_margin_mm=1.0,
        spatial_strategy="fit",
        emit_patch_grid=True,
        patch_shape=(8, 8, 8),
        patch_overlap=0.5,
    )


def test_physical_preprocessing_writes_reusable_provenance_and_complete_patch_grid(tmp_path):
    source = tmp_path / "source.nii.gz"
    _synthetic_ct(source)
    result = preprocess_study("study-1", source, tmp_path / "cache", _spec())

    cached = np.load(result["preprocessed_path"])
    assert cached.shape == (16, 16, 16)
    assert cached.dtype == np.float32
    assert np.isfinite(cached).all()
    assert 0 <= float(cached.min()) <= float(cached.max()) <= 1
    assert validate_cache_entry(result["preprocessed_path"], _spec()) == []

    metadata = json.loads((tmp_path / "cache" / "study-1.npy.metadata.json").read_text())
    assert metadata["preprocessing_implementation"] == PREPROCESSING_IMPLEMENTATION
    assert metadata["orientation"] == "RAS"
    assert metadata["output_shape"] == [16, 16, 16]
    assert metadata["foreground_strategy"] == "external_body"
    assert metadata["crop_fraction"] < 1
    assert len(metadata["source_sha256"]) == 64
    assert np.asarray(metadata["output_affine"]).shape == (4, 4)
    assert metadata["patch_count"] > 1

    coverage = np.zeros((16, 16, 16), dtype=bool)
    with (tmp_path / "cache" / "study-1.npy.patches.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            coverage[
                int(row["z"]): int(row["z"]) + int(row["dz"]),
                int(row["y"]): int(row["y"]) + int(row["dy"]),
                int(row["x"]): int(row["x"]) + int(row["dx"]),
            ] = True
    assert coverage.all()

    reused = preprocess_study("study-1", source, tmp_path / "cache", _spec())
    assert reused["status"] == "cached"


def test_cache_cannot_be_silently_reused_with_another_contract(tmp_path):
    source = tmp_path / "source.nii.gz"
    _synthetic_ct(source)
    preprocess_study("study-1", source, tmp_path / "cache", _spec())

    changed = PreprocessingSpec(**{**_spec().__dict__, "foreground_margin_mm": 2.0})
    with pytest.raises(ValueError, match="different source or preprocessing contract"):
        preprocess_study("study-1", source, tmp_path / "cache", changed)
