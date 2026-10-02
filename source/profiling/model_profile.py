from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from source.components.peft.freeze import trainable_parameter_summary


@contextlib.contextmanager
def _quiet_native_profiler_output():
    """Hide Kineto/USDT diagnostics emitted directly by the native profiler."""
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _leading_batch_size(output: Any) -> int | None:
    """Batch size of the first batched tensor in a forward output (dict/list/tuple/tensor)."""
    if isinstance(output, torch.Tensor):
        return int(output.shape[0]) if output.ndim >= 1 else None
    values = output.values() if isinstance(output, dict) else output if isinstance(output, (list, tuple)) else ()
    for value in values:
        found = _leading_batch_size(value)
        if found:
            return found
    return None


@torch.no_grad()
def profile_model(
    model: nn.Module,
    forward: Callable[[], Any],
    *,
    warmup: int = 2,
    iterations: int = 10,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Latency, FLOPs and peak memory of ``forward()`` in eval mode.

    ``forward()`` runs one batch; the ``*_per_volume`` numbers divide by ``batch_size``
    (inferred from the forward output when not given). Every submodule's train/eval mode is
    restored afterwards, so profiling mid-run does not leave the model in eval.
    """
    summary = trainable_parameter_summary(model)
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    try:
        output = None
        for _ in range(warmup):
            output = forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            output = forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - started) * 1000 / iterations
        peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        gflops: float | None = None
        reason = "PyTorch profiler did not report supported operator FLOPs"
        try:
            with _quiet_native_profiler_output():
                with torch.profiler.profile(with_flops=True) as profiler:
                    output = forward()
            flops = sum(int(event.flops or 0) for event in profiler.key_averages())
            if flops > 0:
                gflops = flops / 1e9
                reason = ""
        except (RuntimeError, NotImplementedError) as exc:
            reason = str(exc)
    finally:
        for module, training in modes:
            module.training = training
    volumes = int(batch_size or _leading_batch_size(output) or 1)
    return {
        **summary,
        "batch_size": volumes,
        "gflops_per_volume": gflops / volumes if gflops is not None else None,
        "gflops_unavailable_reason": reason or None,
        "peak_vram_gb": peak,
        "latency_ms_per_batch": latency,
        "latency_ms_per_volume": latency / volumes,
    }
