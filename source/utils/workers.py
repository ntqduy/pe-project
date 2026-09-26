"""Worker counts that fit the machine the run is on: its CPUs and the RAM free right now.

The same command runs on a 4-vCPU/15 GB VM and a 16-vCPU/64 GB one, and an OOM-killed worker
hangs the parent instead of failing it. So worker settings accept ``auto`` (the default in
the CT-FM wrappers), and an explicit number is treated as an upper bound: when the free
memory cannot hold that many workers, fewer are used and the log line says why.

Per-worker budgets were measured on INSPECT: CT-FM preprocessing of one raw CTPA peaks at
0.8-1.5 GB (246-475 slices), and a DataLoader worker reading one cached tensor stays well
under 0.5 GB.
"""
from __future__ import annotations

import os
from typing import Any

PREPROCESS_WORKER_GB = 2.0     # raw NIfTI -> resampled canvas, plus results in flight
LOADER_WORKER_GB = 0.5         # DataLoader worker reading cached tensors
RESERVE_GB = 4.0               # main process (model, CUDA context) and the OS


def available_memory_gb() -> float | None:
    """MemAvailable from /proc/meminfo (reclaimable cache counts as free), or None."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        return None
    return None


def resolve_workers(
    requested: Any,
    *,
    per_worker_gb: float,
    cpu_margin: int = 0,
    minimum: int = 0,
    maximum: int | None = None,
    share: int = 1,
    reserve_gb: float = RESERVE_GB,
) -> tuple[int, str]:
    """Return ``(workers, explanation)`` for ``requested`` = ``auto``/None/"" or an integer.

    ``share`` splits the machine between processes that each start their own workers (DDP
    ranks). ``minimum`` is 1 for a process pool and 0 for a DataLoader (0 = load in the
    main process). ``explanation`` is meant for the run log.
    """
    cpus = max(1, (os.cpu_count() or 1) // max(1, share))
    cpu_limit = max(minimum, cpus - cpu_margin)
    free = available_memory_gb()
    if free is None:
        ram_limit = cpu_limit
        ram_text = "free RAM unknown"
    else:
        per_process_free = free / max(1, share)
        ram_limit = max(minimum, int((per_process_free - reserve_gb) // per_worker_gb))
        ram_text = f"{free:.1f} GB free RAM"
    basis = (f"{cpus} CPU{'s' if cpus != 1 else ''}{' per rank' if share > 1 else ''}, {ram_text}, "
             f"{per_worker_gb:g} GB/worker + {reserve_gb:g} GB reserve")
    text = str(requested).strip().lower() if requested is not None else ""
    if text in {"", "auto", "none"}:
        workers = max(minimum, min(cpu_limit, ram_limit, maximum if maximum is not None else cpu_limit))
        return workers, f"workers={workers} (auto: {basis})"
    number = int(text)
    if number < minimum:
        raise ValueError(f"worker count must be >= {minimum}, got {number}")
    # An explicit count may exceed the CPUs (I/O-bound loaders), but never the free memory.
    if free is not None and number > ram_limit:
        return ram_limit, f"workers={ram_limit} (requested {number}, capped to avoid OOM: {basis})"
    return number, f"workers={number} (requested: {basis})"
