from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch.distributed as dist

from .setup import DistributedContext


def gather_objects(local: Any, context: DistributedContext) -> list[Any] | None:
    if not context.distributed:
        return [local]
    gathered: list[Any] | None = [None] * context.world_size if context.is_main else None
    dist.gather_object(local, gathered, dst=0)
    return gathered


def gather_prediction_rows(
    local_rows: Sequence[Mapping[str, Any]],
    context: DistributedContext,
    *,
    id_columns: tuple[str, ...] = ("patient_id", "study_id"),
    expected_ids: Iterable[tuple[str, ...]] | None = None,
) -> list[dict[str, Any]] | None:
    shards = gather_objects([dict(row) for row in local_rows], context)
    if not context.is_main:
        return None
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for shard in shards or []:
        for row in shard:
            key = tuple(str(row.get(column, "")) for column in id_columns)
            if not all(key):
                raise ValueError(f"prediction row has empty identifier: {row}")
            if key in merged and merged[key] != row:
                raise ValueError(f"conflicting duplicate distributed prediction: {key}")
            merged.setdefault(key, row)
    if expected_ids is not None:
        expected = {tuple(str(value) for value in key) for key in expected_ids}
        actual = set(merged)
        if actual != expected:
            raise ValueError(f"distributed predictions mismatch: missing={len(expected-actual)} extra={len(actual-expected)}")
    return [merged[key] for key in sorted(merged)]
