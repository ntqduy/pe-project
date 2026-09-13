"""CT geometry and volume-cache statistics.

Reads the sidecars stage 0 already wrote. It deliberately does not open NIfTI volumes:
an EDA pass over 23k scans must stay cheap, and every geometry fact needed here was
already recorded when the cache was built.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .loaders import DatasetBundle, column_names

GEOMETRY_COLUMNS = (
    "shape", "spacing", "orientation", "slice_count",
    "rows", "columns", "slice_thickness", "pixel_spacing",
)


def _numbers(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        candidates = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except (ValueError, TypeError):
            decoded = text.replace("[", "").replace("]", "").split(",")
        candidates = decoded if isinstance(decoded, (list, tuple)) else [decoded]
    output: list[float] = []
    for item in candidates:
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            continue
    return output


def _describe(values: Sequence[float]) -> dict[str, float] | None:
    ordered = sorted(float(v) for v in values if v == v)
    if not ordered:
        return None
    count = len(ordered)
    return {
        "n": count,
        "min": ordered[0],
        "p25": ordered[count // 4],
        "median": ordered[count // 2],
        "p75": ordered[(3 * count) // 4],
        "max": ordered[-1],
        "mean": sum(ordered) / count,
    }


def geometry_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [name for name in GEOMETRY_COLUMNS if name in set(column_names(rows))]
    if not available:
        return {"status": "unavailable", "reason": "manifest carries no geometry column"}
    output: dict[str, Any] = {"status": "available", "columns": available, "axes": {}}
    for column in available:
        per_axis: dict[int, list[float]] = {}
        scalars: list[float] = []
        for row in rows:
            numbers = _numbers(row.get(column))
            if len(numbers) == 1:
                scalars.append(numbers[0])
            for axis, number in enumerate(numbers):
                per_axis.setdefault(axis, []).append(number)
        if len(per_axis) > 1:
            output["axes"][column] = {
                f"axis_{axis}": _describe(values) for axis, values in sorted(per_axis.items())
            }
        elif scalars:
            output["axes"][column] = {"value": _describe(scalars)}
    return output


def cache_summary(dataset_root: Path) -> dict[str, Any]:
    """Size and count of the preprocessed volume cache, if stage 0 wrote one."""
    volumes = dataset_root / "volumes"
    if not volumes.is_dir():
        return {"status": "unavailable", "reason": f"no volume cache at {volumes.resolve()}"}
    files = [path for path in volumes.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    by_suffix: dict[str, int] = {}
    for path in files:
        by_suffix[path.suffix.lower() or "<none>"] = by_suffix.get(path.suffix.lower() or "<none>", 0) + 1
    return {
        "status": "available",
        "path": str(volumes.resolve()),
        "files": len(files),
        "total_bytes": total,
        "total_gib": total / (1024 ** 3),
        "by_suffix": dict(sorted(by_suffix.items())),
    }


def analyse(bundle: DatasetBundle) -> dict[str, Any]:
    rows = bundle.rows("ctpa.csv")
    return {
        "manifest_geometry": geometry_summary(rows) if rows else
            {"status": "unavailable", "reason": "no ctpa.csv"},
        "volume_cache": cache_summary(bundle.dataset_root),
    }
