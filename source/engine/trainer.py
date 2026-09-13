from __future__ import annotations

import csv
import os
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from source.distributed.setup import DistributedContext
from source.engine.checkpoint import save_checkpoint_atomic
from source.utils.logger import RunLogger


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


class Trainer:
    """Small task-agnostic engine; task callbacks own forward/loss/decoding semantics."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_step: Callable[
            [nn.Module, Mapping[str, Any]],
            Tensor | tuple[Tensor, Mapping[str, float | Tensor]],
        ],
        context: DistributedContext,
        run_dir: Path,
        lineage: Mapping[str, Any],
        *,
        scheduler: Any | None = None,
        precision: str = "fp32",
        accumulation_steps: int = 1,
        maximize_metric: bool = True,
        early_stopping_patience: int | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_step = loss_step
        self.context = context
        self.run_dir = run_dir
        self.lineage = dict(lineage)
        self.scheduler = scheduler
        self.precision = precision
        self.accumulation_steps = int(accumulation_steps)
        self.maximize_metric = maximize_metric
        # None disables early stopping and keeps the historical "always run every epoch"
        # behaviour, so a config that does not set it is unaffected.
        self.early_stopping_patience = (
            None if early_stopping_patience is None else int(early_stopping_patience)
        )
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("training.early_stopping_patience must be a positive integer or null")
        if self.accumulation_steps < 1:
            raise ValueError("gradient accumulation must be positive")
        self.autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.amp_enabled = context.device.type == "cuda" and precision in {"bf16", "fp16"}
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and precision == "fp16")
        self.logger = RunLogger(run_dir / "logs" / "run.log", echo=context.is_main) if context.is_main else None

    def _reduce_mean(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.context.device, dtype=torch.float64)
        if self.context.distributed:
            torch.distributed.all_reduce(tensor)
            tensor /= self.context.world_size
        return float(tensor.cpu())

    def _reduce_sum(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.context.device, dtype=torch.float64)
        if self.context.distributed:
            torch.distributed.all_reduce(tensor)
        return float(tensor.cpu())

    @staticmethod
    def _unpack_loss(
        output: Tensor | tuple[Tensor, Mapping[str, float | Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        if isinstance(output, Tensor):
            return output, {}
        loss, metrics = output
        return loss, {
            str(name): float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)
            for name, value in metrics.items()
        }

    def _reduce_epoch_metrics(
        self, totals: Mapping[str, float], occurrences: Mapping[str, int]
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, total in totals.items():
            if name.endswith("valid_count"):
                result[name] = self._reduce_sum(total)
            else:
                local_mean = total / max(occurrences.get(name, 0), 1)
                result[name] = self._reduce_mean(local_mean)
        return result

    def train_epoch(self, loader: Any, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.train()
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        total = 0.0
        batches = 0
        metric_totals: dict[str, float] = {}
        metric_occurrences: dict[str, int] = {}
        self.optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            batch = move_to_device(batch, self.context.device)
            with torch.autocast(self.context.device.type, dtype=self.autocast_dtype, enabled=self.amp_enabled):
                raw_loss, step_metrics = self._unpack_loss(self.loss_step(self.model, batch))
                loss = raw_loss / self.accumulation_steps
            self.scaler.scale(loss).backward()
            if (index + 1) % self.accumulation_steps == 0 or index + 1 == len(loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu()) * self.accumulation_steps
            batches += 1
            for name, value in step_metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value
                metric_occurrences[name] = metric_occurrences.get(name, 0) + 1
        return (
            self._reduce_mean(total / max(batches, 1)),
            self._reduce_epoch_metrics(metric_totals, metric_occurrences),
        )

    @torch.no_grad()
    def validation_loss(self, loader: Any) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total = 0.0
        batches = 0
        metric_totals: dict[str, float] = {}
        metric_occurrences: dict[str, int] = {}
        for batch in loader:
            batch = move_to_device(batch, self.context.device)
            with torch.autocast(self.context.device.type, dtype=self.autocast_dtype, enabled=self.amp_enabled):
                loss, step_metrics = self._unpack_loss(self.loss_step(self.model, batch))
            total += float(loss.detach().cpu())
            batches += 1
            for name, value in step_metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value
                metric_occurrences[name] = metric_occurrences.get(name, 0) + 1
        return (
            self._reduce_mean(total / max(batches, 1)),
            self._reduce_epoch_metrics(metric_totals, metric_occurrences),
        )

    def _write_history(self, rows: list[dict[str, Any]]) -> None:
        destination = self.run_dir / "logs" / "history.csv"
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)

    def fit(
        self,
        train_loader: Any,
        validation_loader: Any,
        epochs: int,
        metric_fn: Callable[[nn.Module, Any, DistributedContext], float] | None = None,
    ) -> dict[str, Any]:
        best = float("-inf") if self.maximize_metric else float("inf")
        history: list[dict[str, Any]] = []
        stalled_epochs = 0
        stopped_early = False
        epochs_run = 0
        started = time.perf_counter()
        for epoch in range(1, int(epochs) + 1):
            epochs_run = epoch
            epoch_started = time.perf_counter()
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            val_loss, validation_metrics = self.validation_loss(validation_loader)
            primary = metric_fn(self.model, validation_loader, self.context) if metric_fn else -val_loss
            if self.scheduler is not None:
                try:
                    self.scheduler.step(primary)
                except TypeError:
                    self.scheduler.step()
            improved = primary > best if self.maximize_metric else primary < best
            peak = torch.cuda.max_memory_allocated(self.context.device) / (1024**3) if self.context.device.type == "cuda" else 0.0
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "primary_val_metric": primary,
                "secondary_val_metric": "",
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time_sec": time.perf_counter() - epoch_started,
                "peak_vram_gb": peak,
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{f"val_{name}": value for name, value in validation_metrics.items()},
            }
            history.append(row)
            if self.context.is_main:
                self._write_history(history)
                self.logger.log(
                    f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"primary={primary:.6f} lr={row['lr']:.6g} time_sec={row['epoch_time_sec']:.2f}"
                )
                if improved:
                    lineage = {**self.lineage, "epoch": epoch, "validation_metric": primary}
                    save_checkpoint_atomic(
                        self.run_dir / "best.ckpt",
                        self.model,
                        lineage=lineage,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )
            if improved:
                best = primary
                stalled_epochs = 0
            else:
                stalled_epochs += 1
            self.context.barrier()
            # Every rank derives `improved` from rank-reduced values, so this stop decision is
            # identical on all ranks. Deciding it per-rank would leave some ranks waiting at
            # the next barrier forever.
            if (
                self.early_stopping_patience is not None
                and stalled_epochs >= self.early_stopping_patience
            ):
                stopped_early = True
                if self.context.is_main and self.logger is not None:
                    self.logger.log(
                        f"early_stopping epoch={epoch} patience={self.early_stopping_patience} "
                        f"epochs_without_improvement={stalled_epochs} best={best:.6f}"
                    )
                break
        return {
            "best_validation_metric": best,
            # `epochs` stays the configured budget; `epochs_run` is what was actually spent.
            "epochs": int(epochs),
            "epochs_run": epochs_run,
            "early_stopping_patience": self.early_stopping_patience,
            "stopped_early": stopped_early,
            "training_time_min": (time.perf_counter() - started) / 60,
            "history": history,
        }
