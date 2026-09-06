from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from collections.abc import Callable

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    initialized_here: bool = False

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(backend: str | None = None, timeout_seconds: int = 1800) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    selected = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    initialized = False
    if world_size > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=selected, init_method="env://", timeout=timedelta(seconds=timeout_seconds))
        initialized = True
    return DistributedContext(rank, local_rank, world_size, selected if world_size > 1 else None, initialized)


def wrap_ddp(model: torch.nn.Module, context: DistributedContext, **kwargs: Any) -> torch.nn.Module:
    model = model.to(context.device)
    if not context.distributed:
        return model
    from torch.nn.parallel import DistributedDataParallel

    if context.device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[context.local_rank], output_device=context.local_rank, **kwargs)
    return DistributedDataParallel(model, **kwargs)


def rank_zero_call(context: DistributedContext, callback: Callable[[], Any]) -> Any:
    """Run a side effect on rank 0 and propagate either its result or failure to every rank."""
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = {"ok": True, "value": callback()}
        except Exception as exc:
            payload[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if context.distributed:
        dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not result["ok"]:
        raise RuntimeError(f"rank-0 operation failed: {result['error']}")
    return result["value"]
