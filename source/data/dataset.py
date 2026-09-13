from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from source.silver.schema import canonical_target, decode_storage_value, target_spec

from .manifests import read_rows


class VolumeLoadError(RuntimeError):
    pass


def load_volume(path: Path) -> Tensor:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".pt", ".pth")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping):
            payload = payload.get("volume", payload.get("image"))
        return torch.as_tensor(payload).float()
    if suffixes.endswith((".npy", ".npz")):
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise VolumeLoadError("NumPy is required for NPY/NPZ volumes") from exc
        payload = np.load(path)
        if hasattr(payload, "files"):
            if len(payload.files) != 1:
                raise VolumeLoadError(f"NPZ must contain exactly one array: {path}")
            payload = payload[payload.files[0]]
        return torch.from_numpy(payload).float()
    if suffixes.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
            import numpy as np
        except ModuleNotFoundError as exc:
            raise VolumeLoadError("nibabel and NumPy are required for NIfTI volumes") from exc
        return torch.from_numpy(np.asarray(nib.load(str(path)).dataobj)).float()
    raise VolumeLoadError(f"unsupported volume format: {path}")


def _float(row: Mapping[str, Any], name: str) -> float:
    raw = row.get(name)
    if raw is None or str(raw).strip() == "":
        return float("nan")
    return float(raw)


def _available(row: Mapping[str, Any], name: str | None) -> bool:
    """Read an explicit modality-level availability flag without imputing it."""
    if not name:
        return False
    raw = row.get(name)
    if raw is None or str(raw).strip() == "":
        return False
    try:
        return float(raw) > 0
    except (TypeError, ValueError):
        return str(raw).strip().upper() in {"TRUE", "T", "YES", "Y"}


class CTPADataset(Dataset[dict[str, Any]]):
    """Dataset API for validated full-data manifests; paths are resolved by the caller."""

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        split: str,
        *,
        transform: Callable[[Tensor], Tensor] | None = None,
        image_column: str = "image_path",
        label_columns: Sequence[str] = (),
        ehr_columns: Sequence[str] = (),
        pesi_columns: Sequence[str] = (),
        ehr_availability_column: str | None = None,
        pesi_availability_column: str | None = None,
        mask_columns: Mapping[str, str] | None = None,
        roi_manifest: str | Path | None = None,
        roi_mask_ids: Mapping[str, str] | None = None,
        roi_control_for: Mapping[str, str] | None = None,
        silver_path: str | Path | None = None,
        silver_targets: Sequence[str] = (),
    ):
        self.manifest = Path(manifest)
        self.data_root = Path(data_root)
        self.rows = [row for row in read_rows(self.manifest) if str(row.get("split")) == split]
        if not self.rows:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.transform = transform
        self.image_column = image_column
        self.label_columns = tuple(label_columns)
        self.ehr_columns = tuple(ehr_columns)
        self.pesi_columns = tuple(pesi_columns)
        self.ehr_availability_column = str(ehr_availability_column or "").strip() or None
        self.pesi_availability_column = str(pesi_availability_column or "").strip() or None
        self.mask_columns = dict(mask_columns or {})
        self.roi_mask_ids = dict(roi_mask_ids or {})
        self.roi_control_for = dict(roi_control_for or {})
        self.roi_lookup: dict[tuple[str, str, str], str] = {}
        if self.roi_mask_ids:
            if roi_manifest is None:
                raise ValueError("data.roi_mask_ids requires data.roi_manifest")
            selected_ids = set(self.roi_mask_ids.values())
            reverse = {roi_id: region for region, roi_id in self.roi_mask_ids.items()}
            if len(reverse) != len(self.roi_mask_ids):
                raise ValueError("each configured ROI ID may supply only one model mask")
            for roi_row in read_rows(roi_manifest):
                roi_id = str(roi_row.get("roi_id") or "")
                if roi_id not in selected_ids:
                    continue
                region = reverse[roi_id]
                expected_control = str(self.roi_control_for.get(region) or "")
                observed_control = str(roi_row.get("control_for") or "")
                if observed_control.lower() == "nan":
                    observed_control = ""
                if observed_control != expected_control:
                    continue
                if str(roi_row.get("status")) not in {"PASS", "SUSPICIOUS", "pass"}:
                    continue
                path = str(roi_row.get("roi_path") or "")
                if not path:
                    continue
                key = (str(roi_row["patient_id"]), str(roi_row["study_id"]), region)
                if key in self.roi_lookup:
                    raise ValueError(f"duplicate ROI mask mapping for {key}")
                self.roi_lookup[key] = path
        self.silver_targets = tuple(silver_targets)
        self.silver_canonical_targets = {
            requested: canonical_target(requested) for requested in self.silver_targets
        }
        self.silver_lookup: dict[tuple[str, str, str], float] = {}
        if silver_path is not None:
            for silver in read_rows(silver_path):
                if str(silver.get("status")) != "accepted":
                    continue
                target = canonical_target(str(silver.get("target")))
                if target not in set(self.silver_canonical_targets.values()):
                    continue
                value = decode_storage_value(silver.get("value"))
                spec = target_spec(target)
                if spec.kind == "categorical":
                    numeric = float(spec.values.index(value)) if value in spec.values else None
                elif spec.kind == "binary":
                    numeric = float(value) if type(value) is bool else None
                else:
                    numeric = (
                        float(value)
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                        else None
                    )
                if numeric is not None:
                    key = (str(silver["patient_id"]), str(silver["study_id"]), target)
                    if key in self.silver_lookup:
                        raise ValueError(
                            f"multiple accepted silver labels map to one study target {key}; "
                            "configure an explicit report-selection step before training"
                        )
                    self.silver_lookup[key] = numeric

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, raw: Any) -> Path:
        path = Path(str(raw))
        return path if path.is_absolute() else self.data_root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve(row[self.image_column])
        volume = load_volume(image_path)
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        if volume.ndim != 4:
            raise VolumeLoadError(f"expected volume [C,D,H,W], got {tuple(volume.shape)} at {image_path}")
        if self.transform:
            volume = self.transform(volume)
        masks: dict[str, Tensor] = {}
        for region, column in self.mask_columns.items():
            mask = load_volume(self._resolve(row[column]))
            masks[region] = mask.unsqueeze(0) if mask.ndim == 3 else mask
        for region in self.roi_mask_ids:
            key = (str(row["patient_id"]), str(row["study_id"]), region)
            if key not in self.roi_lookup:
                raise VolumeLoadError(f"required ROI mask is unavailable: {key}")
            mask = load_volume(self._resolve(self.roi_lookup[key]))
            masks[region] = mask.unsqueeze(0) if mask.ndim == 3 else mask
        labels = torch.tensor([_float(row, name) for name in self.label_columns], dtype=torch.float32)
        item: dict[str, Any] = {
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "volume": volume,
            "masks": masks,
            "labels": labels,
            "label_valid": torch.isfinite(labels),
        }
        if self.ehr_columns:
            item["ehr"] = torch.tensor([_float(row, name) for name in self.ehr_columns], dtype=torch.float32)
            if self.ehr_availability_column:
                item["ehr_available"] = torch.tensor(
                    _available(row, self.ehr_availability_column), dtype=torch.bool
                )
        if self.pesi_columns:
            item["pesi"] = torch.tensor([_float(row, name) for name in self.pesi_columns], dtype=torch.float32)
            if self.pesi_availability_column:
                item["pesi_available"] = torch.tensor(
                    _available(row, self.pesi_availability_column), dtype=torch.bool
                )
        if self.silver_targets:
            key = (str(row["patient_id"]), str(row["study_id"]))
            values = [
                self.silver_lookup.get((*key, self.silver_canonical_targets[target]), float("nan"))
                for target in self.silver_targets
            ]
            item["silver_labels"] = torch.tensor(values, dtype=torch.float32)
            item["silver_valid"] = torch.isfinite(item["silver_labels"])
        return item


class ReportEmbeddingDataset(Dataset[dict[str, Any]]):
    """Report-only dataset that never reads an image or anatomy mask."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        *,
        embedding_columns: Sequence[str],
        label_columns: Sequence[str],
    ):
        self.manifest = Path(manifest)
        self.rows = [row for row in read_rows(self.manifest) if str(row.get("split")) == split]
        if not self.rows:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.embedding_columns = tuple(embedding_columns)
        self.label_columns = tuple(label_columns)
        if not self.embedding_columns:
            raise ValueError("report_embedding_columns is required")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        labels = torch.tensor(
            [_float(row, name) for name in self.label_columns], dtype=torch.float32
        )
        return {
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "report_embedding": torch.tensor(
                [_float(row, name) for name in self.embedding_columns], dtype=torch.float32
            ),
            "labels": labels,
            "label_valid": torch.isfinite(labels),
        }
