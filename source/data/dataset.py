from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from source.silver.schema import canonical_target, decode_storage_value, target_spec

from .manifests import read_rows


class VolumeLoadError(RuntimeError):
    pass


def _contract_value_matches(found: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple)):
        if not isinstance(found, (list, tuple)) or len(found) != len(expected):
            return False
        try:
            return all(math.isclose(float(a), float(b)) for a, b in zip(found, expected, strict=True))
        except (TypeError, ValueError):
            return False
    return str(found).strip().upper() == str(expected).strip().upper()


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


def _observed_label(row: Mapping[str, Any], name: str) -> bool:
    """Return whether a numeric outcome is present and finite in a manifest row."""
    raw = row.get(name)
    if raw is None or str(raw).strip() == "":
        return False
    try:
        return math.isfinite(float(raw))
    except (TypeError, ValueError):
        return False


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
        spesi_columns: Sequence[str] = (),
        ehr_availability_column: str | None = None,
        spesi_availability_column: str | None = None,
        mask_columns: Mapping[str, str] | None = None,
        roi_manifest: str | Path | None = None,
        roi_mask_ids: Mapping[str, str] | None = None,
        roi_control_for: Mapping[str, str] | None = None,
        silver_path: str | Path | None = None,
        silver_targets: Sequence[str] = (),
        drop_rows_without_labels: bool = False,
        input_contract: Mapping[str, Any] | None = None,
        preload: bool = False,
    ):
        self.manifest = Path(manifest)
        # Sidecar fields every input volume must match (e.g. orientation / hu_range for an
        # encoder that converts the cache to its own pre-training contract).
        self.input_contract = dict(input_contract or {})
        self.data_root = Path(data_root)
        self.label_columns = tuple(label_columns)
        source_rows = [row for row in read_rows(self.manifest) if str(row.get("split")) == split]
        self.dropped_unlabeled_rows: tuple[dict[str, Any], ...] = ()
        if drop_rows_without_labels and self.label_columns:
            kept_rows = [
                row for row in source_rows
                if any(_observed_label(row, column) for column in self.label_columns)
            ]
            self.dropped_unlabeled_rows = tuple(
                row for row in source_rows if not any(
                    _observed_label(row, column) for column in self.label_columns
                )
            )
            self.rows = kept_rows
        else:
            self.rows = source_rows
        if not self.rows:
            if self.dropped_unlabeled_rows:
                raise ValueError(
                    f"manifest has no rows with observed labels for split={split!r}; "
                    f"dropped={len(self.dropped_unlabeled_rows)}"
                )
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.transform = transform
        self.image_column = image_column
        self.ehr_columns = tuple(ehr_columns)
        self.spesi_columns = tuple(spesi_columns)
        self.ehr_availability_column = str(ehr_availability_column or "").strip() or None
        self.spesi_availability_column = str(spesi_availability_column or "").strip() or None
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
        # data.preload_inputs: small per-study inputs (pooled CT-FM features, ~2 KB) are read
        # once and kept in RAM; DataLoader workers fork afterwards and inherit them instead of
        # re-reading every file from the bucket each epoch.
        self.memory: dict[str, Tensor] = {}
        self.preload_report: dict[str, Any] = {"enabled": False}
        if preload:
            self.preload_report = self.preload_inputs()

    def preload_inputs(self, threads: int = 16, max_fraction_of_free_ram: float = 0.25) -> dict[str, Any]:
        """Load every row's input tensor into ``self.memory``; skipped when it would not fit."""
        from concurrent.futures import ThreadPoolExecutor

        from source.utils.workers import available_memory_gb

        paths = list(dict.fromkeys(str(self._resolve(row[self.image_column])) for row in self.rows))
        if not paths:
            return {"enabled": True, "loaded": 0}
        first = load_volume(Path(paths[0]))
        estimate_gb = first.numel() * first.element_size() * len(paths) / 1024**3
        free = available_memory_gb()
        if free is not None and estimate_gb > max_fraction_of_free_ram * free:
            return {
                "enabled": True,
                "loaded": 0,
                "skipped": f"{len(paths)} inputs ~{estimate_gb:.1f} GB exceed "
                           f"{max_fraction_of_free_ram:.0%} of {free:.1f} GB free RAM; read per item",
            }
        # Reading from gcsfuse is latency-bound, so threads rather than processes.
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for path, volume in zip(paths[1:], pool.map(lambda path: load_volume(Path(path)), paths[1:])):
                self.memory[path] = volume
        self.memory[paths[0]] = first
        return {"enabled": True, "loaded": len(self.memory), "estimated_gb": round(estimate_gb, 4)}

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, raw: Any) -> Path:
        path = Path(str(raw))
        return path if path.is_absolute() else self.data_root / path

    def _grid_for(self, image_path: Path) -> Any:
        """Grid of a cached model input from its sidecar (None for raw inputs)."""
        from source.imaging.grid import model_grid, read_sidecar

        cache = self.__dict__.setdefault("_grid_cache", {})
        key = str(image_path)
        if key not in cache:
            cache[key] = model_grid(read_sidecar(image_path))
        return cache[key]

    def _check_input_contract(self, image_path: Path) -> None:
        if not self.input_contract:
            return
        checked = self.__dict__.setdefault("_contract_checked", set())
        key = str(image_path)
        if key in checked:
            return
        from source.imaging.grid import read_sidecar

        metadata = read_sidecar(image_path)
        if metadata is None:
            raise VolumeLoadError(
                f"{image_path} has no cache sidecar; the encoder needs inputs with "
                f"{self.input_contract} (build them with the shared volume cache)"
            )
        mismatched = {
            field: (metadata.get(field), expected)
            for field, expected in self.input_contract.items()
            if not _contract_value_matches(metadata.get(field), expected)
        }
        if mismatched:
            details = ", ".join(f"{field}={found!r} (expected {expected!r})" for field, (found, expected) in mismatched.items())
            raise VolumeLoadError(f"{image_path} does not match the encoder input contract: {details}")
        checked.add(key)

    def _load_mask(self, mask_path: Path, image_path: Path, volume: Tensor) -> Tensor:
        """A region mask on the same voxel grid as ``volume``.

        Masks are stored in the original CT geometry while cached inputs are reoriented,
        resampled and cropped; interpolating one onto the other by array shape alone would
        misplace every region. When the input's sidecar describes its grid, the mask is
        mapped by world coordinates (cached once on disk next to the input cache).
        """
        grid = self._grid_for(image_path) if "".join(mask_path.suffixes).lower().endswith((".nii", ".nii.gz")) else None
        if grid is not None:
            from source.imaging.grid import aligned_mask

            array = aligned_mask(mask_path, grid, image_path.parent.parent / "aligned_masks")
            mask = torch.from_numpy(array).float()
        else:
            mask = load_volume(mask_path)
        mask = mask.unsqueeze(0) if mask.ndim == 3 else mask
        if tuple(mask.shape[-3:]) != tuple(volume.shape[-3:]) and not self.__dict__.get("_mask_warned"):
            import warnings

            self.__dict__["_mask_warned"] = True
            warnings.warn(
                f"mask {mask_path.name} shape {tuple(mask.shape[-3:])} differs from the input grid "
                f"{tuple(volume.shape[-3:])} and no sidecar grid is available; it will only be "
                "resized by array shape",
                stacklevel=2,
            )
        return mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve(row[self.image_column])
        self._check_input_contract(image_path)
        cached = self.memory.get(str(image_path)) if self.memory else None
        if cached is None:
            volume = load_volume(image_path)
        else:
            volume = cached.clone() if self.transform else cached   # never let a transform edit the cache
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        if volume.ndim != 4:
            raise VolumeLoadError(f"expected volume [C,D,H,W], got {tuple(volume.shape)} at {image_path}")
        if self.transform:
            volume = self.transform(volume)
        masks: dict[str, Tensor] = {}
        for region, column in self.mask_columns.items():
            masks[region] = self._load_mask(self._resolve(row[column]), image_path, volume)
        for region in self.roi_mask_ids:
            key = (str(row["patient_id"]), str(row["study_id"]), region)
            if key not in self.roi_lookup:
                raise VolumeLoadError(f"required ROI mask is unavailable: {key}")
            masks[region] = self._load_mask(self._resolve(self.roi_lookup[key]), image_path, volume)
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
        if self.spesi_columns:
            item["spesi"] = torch.tensor([_float(row, name) for name in self.spesi_columns], dtype=torch.float32)
            if self.spesi_availability_column:
                item["spesi_available"] = torch.tensor(
                    _available(row, self.spesi_availability_column), dtype=torch.bool
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
