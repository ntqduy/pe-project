from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from source.components.peft.freeze import trainable_parameter_summary


@torch.no_grad()
def profile_model(
    model: nn.Module,
    forward: Callable[[], Any],
    *,
    warmup: int = 2,
    iterations: int = 10,
) -> dict[str, Any]:
    summary = trainable_parameter_summary(model)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    for _ in range(warmup):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = (time.perf_counter() - started) * 1000 / iterations
    peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    gflops: float | None = None
    reason = "PyTorch profiler did not report supported operator FLOPs"
    try:
        with torch.profiler.profile(with_flops=True) as profiler:
            forward()
        flops = sum(int(event.flops or 0) for event in profiler.key_averages())
        if flops > 0:
            gflops = flops / 1e9
            reason = ""
    except (RuntimeError, NotImplementedError) as exc:
        reason = str(exc)
    return {
        **summary,
        "gflops_per_volume": gflops,
        "gflops_unavailable_reason": reason or None,
        "peak_vram_gb": peak,
        "latency_ms_per_volume": latency,
    }
