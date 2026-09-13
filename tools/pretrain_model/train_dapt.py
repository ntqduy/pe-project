from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from source.components.encoders.image.registry import build_image_encoder
from source.components.peft.freeze import apply_peft, trainable_parameter_summary
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.setup import initialize_distributed, rank_zero_call, wrap_ddp
from source.engine.checkpoint import save_checkpoint_atomic
from source.engine.experiment import OutputManager, compact_result
from source.engine.trainer import Trainer
from source.pretraining.dapt.base import build_dapt
from source.utils.console import experiment_header
from source.utils.environment import environment_report
from source.utils.logger import RunLogger
from source.utils.seed import seed_everything

from tools._common import base_parser, build_dataset, build_training_lineage, resolve_cli_config


def main() -> int:
    parser = base_parser("Train in-domain CTPA DAPT (None, MAE, DINO, SimCLR, or Anatomy-DAPT)")
    args = parser.parse_args()
    if args.resume:
        raise SystemExit(
            "--resume is not implemented for DAPT; refusing to merge a fresh run into old output"
        )
    config = resolve_cli_config(args)
    context = initialize_distributed(str(config["compute"].get("distributed_backend") or "") or None)
    try:
        paths = ProjectPaths.resolve(config)
        preflight = rank_zero_call(context, lambda: require_preflight(config, paths))
        manager = OutputManager(paths)
        experiment_id = str(config["experiment"]["id"])
        run_dir = manager.run_dir("dapt", experiment_id)

        def prepare() -> Path:
            destination = manager.prepare(
                "dapt",
                experiment_id,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            manager.write_config(destination, {**config, "resolved_paths": paths.as_dict()})
            return destination

        run_dir = rank_zero_call(context, prepare)
        if context.is_main:
            print(experiment_header(config, run_dir, preflight.manifest))
        seed_everything(int(config["seed"]) + context.rank)
        model_config = {**dict(config.get("model") or {}), "data_mode": str(config["data"]["mode"])}
        encoder = build_image_encoder(model_config)
        peft_report = apply_peft(encoder, dict(config.get("peft") or {"method": "full"}))
        method = str((config.get("dapt") or {}).get("method", "none"))
        objective = build_dapt(method, encoder, config.get("dapt"))
        objective = wrap_ddp(objective, context)
        reproducibility = environment_report(paths.code_root)
        lineage = build_training_lineage(
            config,
            paths,
            code_commit=reproducibility["git_commit"],
            source_checkpoint=config["model"].get("checkpoint"),
            overrides={"dapt": method},
        )
        if method.lower() in {"none", "d00"}:
            if context.is_main:
                RunLogger(run_dir / "logs" / "train.log", echo=False).log(
                    "D00 none: normalized public initialization; no DAPT optimization"
                )
                save_checkpoint_atomic(
                    run_dir / "best.ckpt",
                    objective,
                    lineage={**lineage, "epoch": 0, "validation_metric": None},
                )
                result = compact_result(
                    config,
                    status="completed",
                    data={"split_patients": (preflight.manifest or {}).get("split_patients", {})},
                    model={
                        **trainable_parameter_summary(
                            objective.module if hasattr(objective, "module") else objective
                        ),
                        "peft": peft_report,
                    },
                    compute={
                        "strategy": config["compute"]["strategy"],
                        "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                    },
                    evaluation={"method": "none", "best_validation_metric": None},
                    reproducibility=reproducibility,
                )
                manager.write_result(run_dir, result)
            context.barrier()
            return 0
        train_data = build_dataset(config, paths, "train")
        validation_data = build_dataset(config, paths, "validation")
        batch_size = int((config.get("training") or {}).get("batch_size", 1))
        workers = int(config["compute"].get("num_workers", 0))
        train_sampler = DistributedSampler(train_data, shuffle=True) if context.distributed else None
        validation_sampler = (
            DistributedSampler(validation_data, shuffle=False) if context.distributed else None
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=workers,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            sampler=validation_sampler,
            num_workers=workers,
        )

        def loss_step(model, batch):
            target = model.module if hasattr(model, "module") else model
            volume = batch["volume"]
            normalized = method.lower()
            if normalized in {"mae", "d01"}:
                return model(volume)["loss"]
            if normalized in {"simclr", "d03"}:
                noise_std = float((config.get("dapt") or {}).get("noise_std", 0.02))
                first_view = volume + torch.randn_like(volume) * noise_std
                second_view = volume + torch.randn_like(volume) * noise_std
                return model(first_view, second_view)["loss"]
            if normalized in {"dino", "d02"}:
                target.update_teacher()
                return model(volume, volume + torch.randn_like(volume) * 0.02)["loss"]
            region = str((config.get("dapt") or {}).get("anatomy_region", "pa"))
            return model(volume, batch["masks"][region])["loss"]

        parameters = [parameter for parameter in objective.parameters() if parameter.requires_grad]
        training = dict(config.get("training") or {})
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        trainer = Trainer(
            objective,
            optimizer,
            loss_step,
            context,
            run_dir,
            lineage,
            precision=str(config["compute"].get("precision", "fp32")),
            early_stopping_patience=training.get("early_stopping_patience"),
            accumulation_steps=int(training.get("gradient_accumulation", 1)),
        )
        training_result = trainer.fit(train_loader, validation_loader, int(training.get("epochs", 1)))
        if context.is_main:
            result = compact_result(
                config,
                status="completed",
                data={"split_patients": (preflight.manifest or {}).get("split_patients", {})},
                model={
                    **trainable_parameter_summary(
                        objective.module if hasattr(objective, "module") else objective
                    ),
                    "peft": peft_report,
                },
                compute={
                    "strategy": config["compute"]["strategy"],
                    "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                    "training_time_min": training_result["training_time_min"],
                },
                evaluation={
                    "method": method,
                    "best_validation_metric": training_result["best_validation_metric"],
                },
                reproducibility=reproducibility,
            )
            manager.write_result(run_dir, result)
            print(json.dumps({"status": "completed", "output": str(run_dir)}, indent=2))
        context.barrier()
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
