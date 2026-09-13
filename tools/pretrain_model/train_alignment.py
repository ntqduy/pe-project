from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from source.components.encoders.image.registry import build_image_encoder
from source.components.peft.freeze import apply_peft, trainable_parameter_summary
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.setup import initialize_distributed, rank_zero_call, wrap_ddp
from source.engine.experiment import OutputManager, compact_result
from source.engine.trainer import Trainer
from source.engine.transfer import transfer_modules
from source.pretraining.alignment.image_report import ImageReportAlignment
from source.utils.console import experiment_header
from source.utils.environment import environment_report
from source.utils.seed import seed_everything

from tools._common import (
    PrecomputedReportDataset,
    base_parser,
    build_dataset,
    build_training_lineage,
    resolve_cli_config,
)


class IdentityReportEncoder(nn.Module):
    def forward(self, values):
        return values


def main() -> int:
    parser = base_parser("Train paired CTPA-report alignment and produce C0")
    args = parser.parse_args()
    if args.resume:
        raise SystemExit(
            "--resume is not implemented for alignment; "
            "refusing to merge a fresh run into old output"
        )
    config = resolve_cli_config(args)
    context = initialize_distributed(str(config["compute"].get("distributed_backend") or "") or None)
    try:
        paths = ProjectPaths.resolve(config)
        preflight = rank_zero_call(context, lambda: require_preflight(config, paths))
        manager = OutputManager(paths)
        experiment_id = str(config["experiment"]["id"])
        run_dir = manager.run_dir("alignment", experiment_id)

        def prepare() -> Path:
            destination = manager.prepare(
                "alignment",
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
        alignment = dict(config.get("alignment") or {})
        source_checkpoint = (config.get("lineage") or {}).get("source_checkpoint")
        model = ImageReportAlignment(
            encoder,
            IdentityReportEncoder(),
            report_dim=len(alignment.get("report_embedding_columns") or ()),
            projection_dim=int(alignment.get("projection_dim", 256)),
            temperature=float(alignment.get("temperature", 0.07)),
        )
        if source_checkpoint:
            transfer_modules(model, source_checkpoint, ("image_encoder",))
        model = wrap_ddp(model, context)
        embedding_columns = alignment.get("report_embedding_columns")
        train_data = PrecomputedReportDataset(
            build_dataset(config, paths, "train"),
            embedding_columns,
        )
        validation_data = PrecomputedReportDataset(
            build_dataset(config, paths, "validation"),
            embedding_columns,
        )
        batch_size = int((config.get("training") or {}).get("batch_size", 1))
        if batch_size < 2:
            raise ValueError("contrastive alignment requires batch_size >= 2")
        train_sampler = DistributedSampler(train_data, shuffle=True) if context.distributed else None
        validation_sampler = (
            DistributedSampler(validation_data, shuffle=False) if context.distributed else None
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            sampler=validation_sampler,
        )
        training = dict(config.get("training") or {})
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(training.get("learning_rate", 1e-4)),
        )
        reproducibility = environment_report(paths.code_root)
        lineage = build_training_lineage(
            config,
            paths,
            code_commit=reproducibility["git_commit"],
            source_checkpoint=source_checkpoint,
            overrides={"alignment": True},
        )
        trainer = Trainer(
            model,
            optimizer,
            lambda current, batch: current(batch["volume"], batch["report_embedding"])["loss"],
            context,
            run_dir,
            lineage,
            precision=str(config["compute"].get("precision", "fp32")),
            early_stopping_patience=training.get("early_stopping_patience"),
        )
        training_result = trainer.fit(train_loader, validation_loader, int(training.get("epochs", 1)))
        if context.is_main:
            manager.write_result(
                run_dir,
                compact_result(
                    config,
                    status="completed",
                    data={"split_patients": (preflight.manifest or {}).get("split_patients", {})},
                    model={
                        **trainable_parameter_summary(
                            model.module if hasattr(model, "module") else model
                        ),
                        "peft": peft_report,
                        "report_encoder": alignment.get("report_model_id", "precomputed"),
                    },
                    compute={
                        "strategy": config["compute"]["strategy"],
                        "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                        "training_time_min": training_result["training_time_min"],
                    },
                    evaluation={
                        "objective": "symmetric_infonce",
                        "best_validation_metric": training_result["best_validation_metric"],
                    },
                    reproducibility=reproducibility,
                ),
            )
            print(json.dumps({"status": "completed", "output": str(run_dir)}, indent=2))
        context.barrier()
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
