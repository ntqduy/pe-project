from __future__ import annotations

import csv
import math
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
from source.utils.progress import PROGRESS_EVERY_SEC, format_duration, with_progress


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


def resolve_precision(precision: str, device: torch.device) -> str:
    """``auto``: bf16 where the GPU supports it, fp16 (+ GradScaler) on older GPUs, fp32 on CPU."""
    precision = str(precision or "fp32")
    if precision != "auto":
        return precision
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


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
        selection_metric: str | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_step = loss_step
        self.context = context
        self.run_dir = run_dir
        self.lineage = dict(lineage)
        self.scheduler = scheduler
        self.precision = resolve_precision(precision, context.device)
        # A key of the epoch metrics (e.g. "val_auroc") that selects best.ckpt and drives early
        # stopping; None keeps the metric_fn / negative-validation-loss selection.
        self.selection_metric = selection_metric
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
        self.autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.amp_enabled = context.device.type == "cuda" and self.precision in {"bf16", "fp16"}
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.precision == "fp16")
        self.logger = RunLogger(run_dir / "logs" / "run.log", echo=context.is_main) if context.is_main else None
        self._epochs = 0      # set by fit(); only used to label progress lines

    @property
    def _progress_log(self) -> Callable[[str], None] | None:
        return self.logger.log if self.logger is not None else None

    def _reduce_mean(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.context.device, dtype=torch.float64)
        if self.context.distributed:
            torch.distributed.all_reduce(tensor)
            tensor /= self.context.world_size
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
        # A rank can lack a key another rank logged (losses.py skips a target's loss when
        # its batch has no valid label), so every rank reduces the same sorted union of
        # names in one collective; a per-key all_reduce in each rank's own order would
        # pair different metrics across ranks or hang.
        names = set(totals)
        if self.context.distributed:
            gathered: list[Any] = [None] * self.context.world_size
            torch.distributed.all_gather_object(gathered, sorted(names))
            names = set().union(*(set(item or ()) for item in gathered))
        ordered = sorted(names)
        if not ordered:
            return {}
        stacked = torch.tensor(
            [
                [float(totals.get(name, 0.0)) for name in ordered],
                [float(occurrences.get(name, 0)) for name in ordered],
            ],
            device=self.context.device,
            dtype=torch.float64,
        )
        if self.context.distributed:
            torch.distributed.all_reduce(stacked)
        sums, counts = stacked.cpu().tolist()
        result: dict[str, float] = {}
        for name, total, count in zip(ordered, sums, counts):
            if name.endswith("valid_count"):
                result[name] = total
            else:
                # Mean over every step that logged the metric, on any rank.
                result[name] = total / count if count > 0 else float("nan")
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
        progress = with_progress(
            loader, self._progress_log, f"epoch {epoch}/{self._epochs or '?'} train",
            extra=lambda: f" loss={total / max(batches, 1):.4f}",
        )
        for index, batch in enumerate(progress):
            batch = move_to_device(batch, self.context.device)
            # The last group can be shorter than accumulation_steps; divide by its real size
            # so that update averages over the batches it actually holds.
            group_start = (index // self.accumulation_steps) * self.accumulation_steps
            group_size = min(self.accumulation_steps, len(loader) - group_start)
            with torch.autocast(self.context.device.type, dtype=self.autocast_dtype, enabled=self.amp_enabled):
                raw_loss, step_metrics = self._unpack_loss(self.loss_step(self.model, batch))
                loss = raw_loss / group_size
            self.scaler.scale(loss).backward()
            if (index + 1) % self.accumulation_steps == 0 or index + 1 == len(loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            total += float(raw_loss.detach().cpu())
            batches += 1
            for name, value in step_metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value
                metric_occurrences[name] = metric_occurrences.get(name, 0) + 1
        return (
            self._reduce_mean(total / max(batches, 1)),
            self._reduce_epoch_metrics(metric_totals, metric_occurrences),
        )

    @torch.no_grad()
    def validation_loss(self, loader: Any, label: str = "validation") -> tuple[float, dict[str, float]]:
        self.model.eval()
        total = 0.0
        batches = 0
        metric_totals: dict[str, float] = {}
        metric_occurrences: dict[str, int] = {}
        for batch in with_progress(loader, self._progress_log, label):
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
            # Union in first-seen order: a later epoch can log a metric the first one lacked.
            fieldnames = list(dict.fromkeys(name for row in rows for name in row))
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)

    @staticmethod
    def _epoch_line(
        epoch: int,
        epochs: int,
        row: Mapping[str, Any],
        epoch_metrics: Mapping[str, Any],
        improved: bool,
        custom_selection: bool,
    ) -> str:
        """One readable log line per epoch: losses, per-target train/val AUROC, best marker."""
        line = (
            f"epoch={epoch}/{epochs} train_loss={float(row['train_loss']):.6f} "
            f"val_loss={float(row['val_loss']):.6f}"
        )
        if custom_selection:
            # With the default selection the metric is just -val_loss; print it only when custom.
            line += f" selection_metric={float(row['primary_val_metric']):.6f}"
        line += f" lr={float(row['lr']):.6g} time_sec={float(row['epoch_time_sec']):.2f}"
        targets = [
            name.removeprefix("train_").removesuffix("_auroc")
            for name in epoch_metrics
            if name.startswith("train_") and name.endswith("_auroc") and name != "train_auroc"
        ]
        if not targets and ("train_auroc" in epoch_metrics or "val_auroc" in epoch_metrics):
            targets = [""]

        def auroc(value: Any) -> str:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return "nan"
            return f"{number:.4f}" if math.isfinite(number) else "nan"

        if targets:
            parts = []
            for target in targets:
                prefix = f"_{target}" if target else ""
                parts.append(
                    f"{target or 'primary'}={auroc(epoch_metrics.get(f'train{prefix}_auroc'))}"
                    f"/{auroc(epoch_metrics.get(f'val{prefix}_auroc'))}"
                )
            line += " auroc(train/val): " + " ".join(parts)
        if improved:
            line += " | saved best.ckpt"
        return line

    def fit(
        self,
        train_loader: Any,
        validation_loader: Any,
        epochs: int,
        metric_fn: Callable[[nn.Module, Any, DistributedContext], float] | None = None,
        epoch_metrics_fn: Callable[
            [nn.Module, Any, Any, DistributedContext], Mapping[str, float]
        ]
        | None = None,
    ) -> dict[str, Any]:
        best = float("-inf") if self.maximize_metric else float("inf")
        history: list[dict[str, Any]] = []
        stalled_epochs = 0
        stopped_early = False
        epochs_run = 0
        started = time.perf_counter()
        self._epochs = int(epochs)
        for epoch in range(1, int(epochs) + 1):
            epochs_run = epoch
            epoch_started = time.perf_counter()
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            val_loss, validation_metrics = self.validation_loss(
                validation_loader, label=f"epoch {epoch}/{int(epochs)} validation"
            )
            epoch_metrics = (
                dict(epoch_metrics_fn(self.model, train_loader, validation_loader, self.context))
                if epoch_metrics_fn
                else {}
            )
            if self.selection_metric:
                if self.selection_metric not in epoch_metrics:
                    raise KeyError(
                        f"selection metric {self.selection_metric!r} is not among the epoch metrics "
                        f"{sorted(epoch_metrics)}"
                    )
                # NaN (e.g. a single-class validation split) never counts as an improvement.
                primary = float(epoch_metrics[self.selection_metric])
            elif metric_fn:
                primary = metric_fn(self.model, validation_loader, self.context)
            else:
                primary = -val_loss
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
                **epoch_metrics,
            }
            history.append(row)
            if self.context.is_main:
                self._write_history(history)
                self.logger.log(self._epoch_line(
                    epoch, int(epochs), row, epoch_metrics, improved,
                    custom_selection=metric_fn is not None or bool(self.selection_metric),
                ))
                per_epoch = (time.perf_counter() - started) / epoch
                if epoch < int(epochs) and per_epoch >= PROGRESS_EVERY_SEC:    # silent on short smoke epochs
                    self.logger.log(
                        f"run progress: {epoch}/{int(epochs)} epochs in {format_duration(time.perf_counter() - started)}, "
                        f"{format_duration(per_epoch)}/epoch, at most ~{format_duration(per_epoch * (int(epochs) - epoch))} "
                        f"left (early stopping patience {self.early_stopping_patience})"
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
        if self.context.is_main and history:
            # Keep the final state separately from the validation-selected state.  Writing it
            # once avoids a large checkpoint I/O cost on every epoch.
            final_row = history[-1]
            save_checkpoint_atomic(
                self.run_dir / "last.ckpt",
                self.model,
                lineage={
                    **self.lineage,
                    "epoch": int(final_row["epoch"]),
                    "validation_metric": float(final_row["primary_val_metric"]),
                },
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
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
