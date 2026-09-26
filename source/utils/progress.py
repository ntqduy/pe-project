"""Periodic progress lines for long passes over a DataLoader.

A full-cohort epoch is ~19k batches of one study, so a line per batch (or per 25) would bury
the log; instead ``with_progress`` logs at most once per ``every_sec`` (default 60 s, or
``PE_PROGRESS_EVERY_SEC``) and stays silent for passes shorter than that, so smoke runs look
exactly as before.

    for batch in with_progress(loader, log, "epoch 3/50 train"):
        ...
    # -> "epoch 3/50 train 2410/18900 batches (31.2 batches/s, ~8.8 min left)"
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

Item = TypeVar("Item")

PROGRESS_EVERY_SEC = float(os.environ.get("PE_PROGRESS_EVERY_SEC") or 60.0)


def format_duration(seconds: float) -> str:
    """Compact duration for log lines: 45 s, 12.5 min, 3.2 h."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def with_progress(
    items: Iterable[Item],
    log: Callable[[str], None] | None,
    label: str,
    *,
    total: int | None = None,
    unit: str = "batches",
    every_sec: float | None = None,
    extra: Callable[[], str] | None = None,
) -> Iterator[Item]:
    """Yield ``items`` unchanged, logging ``label done/total (rate, ~left)`` every ``every_sec``.

    ``log=None`` (a non-main rank, or no logger) makes this a plain pass-through. ``extra``
    returns text appended to each line, for example a running loss.
    """
    if log is None:
        yield from items
        return
    every_sec = PROGRESS_EVERY_SEC if every_sec is None else every_sec
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None
    began = last = time.perf_counter()
    done = 0
    for item in items:
        yield item
        done += 1
        now = time.perf_counter()
        if now - last < every_sec or (total is not None and done >= total):
            continue
        last = now
        rate = done / (now - began)
        remaining = f", ~{format_duration((total - done) / rate)} left" if total else ""
        suffix = extra() if extra else ""
        log(f"{label} {done}/{total if total is not None else '?'} {unit} ({rate:.1f} {unit}/s{remaining}){suffix}")
