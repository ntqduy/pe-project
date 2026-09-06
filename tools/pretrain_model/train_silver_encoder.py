from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from source.components.encoders.image.registry import build_image_encoder
from source.components.peft.freeze import apply_peft, trainable_parameter_summary
from source.data.manifests import read_rows
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.gather import gather_objects
from source.distributed.setup import initialize_distributed, rank_zero_call, wrap_ddp
from source.engine.checkpoint import load_checkpoint
from source.engine.experiment import OutputManager, compact_result
from source.engine.trainer import Trainer, move_to_device
from source.engine.transfer import transfer_modules
from source.metrics.silver import silver_validation_metrics
from source.pretraining.silver_supervised import (
    TARGET_CLASS_COUNTS,
    SilverEncoderAdaptationModel,
    silver_adaptation_loss,
    summarize_silver_labels,
)
from source.utils.config import validate_config
from source.utils.environment import environment_report
from source.utils.seed import seed_everything
from tools._common import (
    base_parser,
    build_dataset,
    build_training_lineage,
    resolve_cli_config,
    resolve_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_contract(config: Mapping[str, Any]) -> dict[str, int]:
    section = dict(config.get("silver_training") or {})
    raw = section.get("targets") or {}
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("silver_training.targets must map target names to class counts")
    targets = {str(name): int(classes) for name, classes in raw.items()}
    unknown = set(targets) - set(TARGET_CLASS_COUNTS)
    invalid = {
        name: classes
        for name, classes in targets.items()
        if name in TARGET_CLASS_COUNTS and classes != TARGET_CLASS_COUNTS[name]
    }
    if unknown or invalid:
        raise ValueError(f"invalid silver targets: unknown={sorted(unknown)} class_counts={invalid}")
    return targets


def _loss_weights(config: Mapping[str, Any], targets: Mapping[str, int]) -> dict[str, float]:
    configured = dict((config.get("silver_training") or {}).get("loss_weights") or {})
    unknown = set(configured) - set(targets)
    if unknown:
        raise ValueError(f"loss weights configured for unknown targets: {sorted(unknown)}")
    weights = {name: float(configured.get(name, 1.0)) for name in targets}
    if any(value < 0 for value in weights.values()) or not any(value > 0 for value in weights.values()):
        raise ValueError("target loss weights must be non-negative with at least one positive value")
    return weights


def _runtime_config(config: dict[str, Any]) -> tuple[dict[str, Any], ProjectPaths, Path, Path]:
    paths = ProjectPaths.resolve(config)
    section = dict(config.get("silver_training") or {})
    source = str(section.get("silver_source") or "").upper()
    sources = dict(section.get("sources") or {})
    if source not in {"SL00", "SL01", "SL02"} or source not in sources:
        raise ValueError("silver_training.silver_source must select configured SL00, SL01, or SL02")
    silver_path = paths.output_asset(sources[source])
    if silver_path is None:
        raise ValueError(f"silver source path is not configured: {source}")
    initialization = dict(config.get("init") or {})
    allowed_encoders = {"C0", "C_rsna_single", "C_rsna_multi"}
    encoder_label = str(initialization.get("encoder") or "").strip()
    if encoder_label not in allowed_encoders:
        raise ValueError(
            "init.encoder must explicitly identify the source image encoder "
            f"({sorted(allowed_encoders)}) for silver encoder adaptation"
        )
    checkpoint = paths.output_asset(initialization.get("checkpoint"))
    if checkpoint is None:
        raise ValueError("init.checkpoint must explicitly identify the source encoder checkpoint")
    config["silver_training"] = {**section, "silver_source": source}
    config["supervision"] = {
        **dict(config.get("supervision") or {}),
        "require_silver_labels": True,
        "silver_labels": str(silver_path),
    }
    config["lineage"] = {
        **dict(config.get("lineage") or {}),
        "initialization": encoder_label,
        "source_experiment": initialization.get("experiment"),
        "source_checkpoint": str(checkpoint),
        "silver_method": source,
    }
    config = validate_config({key: value for key, value in config.items() if key != "config_hash"})
    return config, ProjectPaths.resolve(config), checkpoint, silver_path


def _accepted_subset(dataset: Dataset, target_names: tuple[str, ...]) -> Subset:
    rows = getattr(dataset, "rows", None)
    lookup = getattr(dataset, "silver_lookup", None)
    if rows is None or lookup is None:
        raise TypeError("silver encoder adaptation requires CTPADataset silver lookup support")
    indices = []
    for index, row in enumerate(rows):
        patient_id = str(row["patient_id"])
        study_id = str(row["study_id"])
        if any((patient_id, study_id, target) in lookup for target in target_names):
            indices.append(index)
    if not indices:
        raise ValueError("split contains no studies with accepted silver targets")
    return Subset(dataset, indices)


def _assert_encoder_compatible(report: Mapping[str, Any]) -> None:
    missing = [key for key in report.get("missing_keys", []) if str(key).startswith("image_encoder.")]
    unexpected = [
        key for key in report.get("unexpected_keys", []) if str(key).startswith("image_encoder.")
    ]
    if missing or unexpected:
        raise RuntimeError(
            "C0 encoder architecture is incompatible: "
            f"missing_encoder_keys={missing} unexpected_encoder_keys={unexpected}"
        )


def _header(
    *,
    checkpoint: Path,
    source: str,
    statistics: Mapping[str, Any],
    parameter_report: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
    encoder_label: str = "C0",
) -> str:
    studies = statistics["studies"]
    lines = [
        "=" * 60,
        "SILVER ENCODER ADAPTATION",
        "=" * 60,
        f"Init encoder              : {encoder_label}",
        f"Init checkpoint           : {checkpoint}",
        "Loaded successfully       : YES",
        "",
        f"Silver source             : {source}",
        f"Studies in train          : {studies['train']['with_accepted_target']}",
        f"Studies in validation     : {studies['validation']['with_accepted_target']}",
        "",
        "Silver targets",
        f"Eligible                  : {statistics['eligible']}",
        f"Accepted                  : {statistics['accepted']}",
        f"Abstained                 : {statistics['abstained']}",
        f"No result                 : {statistics['no_result']}",
        f"Effective coverage        : {100 * float(statistics['effective_coverage']):.2f}%",
        "",
        "Per-target accepted:",
    ]
    for target, counts in statistics["per_target"].items():
        lines.append(f"{target:26s}: {counts['accepted']}")
    compute = dict(config.get("compute") or {})
    lines.extend(
        (
            "",
            "Encoder",
            f"Total params              : {parameter_report['total_params']}",
            f"Trainable params          : {parameter_report['trainable_params']}",
            f"Trainable %               : {float(parameter_report['trainable_percent']):.4f}",
            "",
            f"GPU(s)                    : {','.join(map(str, compute.get('devices') or ())) or 'CPU'}",
            f"Precision                 : {compute.get('precision')}",
            "",
            f"Output                    : {output}",
            "=" * 60,
        )
    )
    return "\n".join(lines)


@torch.no_grad()
def _local_validation_rows(model, loader, context, target_names: tuple[str, ...]) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_to_device(batch, context.device)
        output = model(batch["volume"])
        logits = output["logits"]
        for index, target in enumerate(target_names):
            valid = batch["silver_valid"][:, index].bool()
            if not bool(valid.any()):
                continue
            prediction = logits[target]
            probabilities = (
                torch.sigmoid(prediction)
                if prediction.shape[-1] == 1
                else torch.softmax(prediction, dim=-1)
            )
            selected = valid.nonzero(as_tuple=False).flatten().tolist()
            for batch_index in selected:
                rows.append(
                    {
                        "patient_id": str(batch["patient_id"][batch_index]),
                        "study_id": str(batch["study_id"][batch_index]),
                        "target": target,
                        "y_true": int(batch["silver_labels"][batch_index, index].item()),
                        "scores": [float(value) for value in probabilities[batch_index].detach().cpu()],
                    }
                )
    return rows


def _gather_validation_rows(local_rows, context) -> list[dict[str, Any]] | None:
    shards = gather_objects(local_rows, context)
    if not context.is_main:
        return None
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for shard in shards or []:
        for row in shard:
            key = (str(row["patient_id"]), str(row["study_id"]), str(row["target"]))
            if key in merged and merged[key] != row:
                raise ValueError(f"conflicting distributed silver validation prediction: {key}")
            merged.setdefault(key, dict(row))
    return [merged[key] for key in sorted(merged)]


def main() -> int:
    parser = base_parser("Adapt the shared C0 image encoder using accepted silver labels")
    args = parser.parse_args()
    if args.resume:
        raise SystemExit(
            "--resume is not implemented for silver encoder adaptation; "
            "refusing to merge a fresh run into old output"
        )
    config, paths, init_checkpoint, silver_path = _runtime_config(resolve_cli_config(args))
    context = initialize_distributed(str(config["compute"].get("distributed_backend") or "") or None)
    try:
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"explicit C0 checkpoint not found: {init_checkpoint}")
        if not silver_path.is_file():
            raise FileNotFoundError(f"offline silver labels not found: {silver_path}")
        targets = _target_contract(config)
        weights = _loss_weights(config, targets)
        manifest_rows = read_rows(resolve_manifest(config, paths))
        statistics = summarize_silver_labels(read_rows(silver_path), manifest_rows, tuple(targets))
        available = set(statistics["available_targets"])
        targets = {name: classes for name, classes in targets.items() if name in available}
        weights = {name: weights[name] for name in targets}
        if not targets:
            raise ValueError("selected silver source has no configured targets on train/validation splits")
        config["silver_training"]["targets"] = targets
        config["silver_training"]["loss_weights"] = weights
        config = validate_config({key: value for key, value in config.items() if key != "config_hash"})
        paths = ProjectPaths.resolve(config)
        preflight = rank_zero_call(context, lambda: require_preflight(config, paths))
        manager = OutputManager(paths)
        experiment_id = str(config["experiment"]["id"])
        planned_run_dir = manager.run_dir("silver_encoder_adaptation", experiment_id)
        if init_checkpoint == planned_run_dir or planned_run_dir in init_checkpoint.parents:
            raise ValueError("C0 checkpoint cannot be inside the C_silver output directory")

        def prepare() -> Path:
            destination = manager.prepare(
                "silver_encoder_adaptation",
                experiment_id,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            manager.write_config(destination, {**config, "resolved_paths": paths.as_dict()})
            return destination

        run_dir = rank_zero_call(context, prepare)
        seed_everything(int(config["seed"]) + context.rank)
        init_checksum = _sha256(init_checkpoint)
        model_config = {**dict(config.get("model") or {}), "data_mode": str(config["data"]["mode"])}
        model = SilverEncoderAdaptationModel(build_image_encoder(model_config), targets)
        transfer_report = transfer_modules(model, init_checkpoint, ("image_encoder",))
        _assert_encoder_compatible(transfer_report)
        configured_init_experiment = (config.get("init") or {}).get("experiment")
        checkpoint_init_experiment = (transfer_report.get("lineage") or {}).get("experiment_id")
        if configured_init_experiment and checkpoint_init_experiment != configured_init_experiment:
            raise RuntimeError(
                "C0 checkpoint lineage mismatch: "
                f"configured={configured_init_experiment} checkpoint={checkpoint_init_experiment}"
            )
        peft_report = apply_peft(model.image_encoder, dict(config.get("peft") or {"method": "full"}))
        encoder_parameter_report = trainable_parameter_summary(model.image_encoder)
        if context.is_main:
            print(
                _header(
                    checkpoint=init_checkpoint,
                    source=str(config["silver_training"]["silver_source"]),
                    statistics=statistics,
                    parameter_report=encoder_parameter_report,
                    config=config,
                    output=run_dir,
                    encoder_label=str((config.get("lineage") or {}).get("initialization", "C0")),
                )
            )

        target_names = tuple(targets)
        train_data = _accepted_subset(build_dataset(config, paths, "train"), target_names)
        validation_data = _accepted_subset(build_dataset(config, paths, "validation"), target_names)
        workers = int(config["compute"].get("num_workers", 0))
        batch_size = int((config.get("training") or {}).get("batch_size", 1))
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
            shuffle=False,
            sampler=validation_sampler,
            num_workers=workers,
        )
        model = wrap_ddp(model, context)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("silver encoder adaptation has no trainable parameters")
        training = dict(config.get("training") or {})
        epochs = int(training.get("epochs", 0))
        if epochs < 1:
            raise ValueError("silver encoder adaptation requires training.epochs >= 1")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )

        def loss_step(current, batch):
            output = current(batch["volume"])
            return silver_adaptation_loss(
                output["logits"],
                batch["silver_labels"],
                batch["silver_valid"],
                target_names,
                weights,
            )

        reproducibility = environment_report(paths.code_root)
        lineage = build_training_lineage(
            config,
            paths,
            code_commit=reproducibility["git_commit"],
            source_checkpoint=init_checkpoint,
            overrides={
                "source_experiment": (config.get("init") or {}).get("experiment"),
                "silver_source": config["silver_training"]["silver_source"],
            },
        )
        trainer = Trainer(
            model,
            optimizer,
            loss_step,
            context,
            run_dir,
            lineage,
            precision=str(config["compute"].get("precision", "fp32")),
            accumulation_steps=int(training.get("gradient_accumulation", 1)),
        )
        training_result = trainer.fit(train_loader, validation_loader, epochs)
        context.barrier()
        load_checkpoint(run_dir / "best.ckpt", model=model, strict=True)
        local_rows = _local_validation_rows(model, validation_loader, context, target_names)
        validation_rows = _gather_validation_rows(local_rows, context)

        def finalize() -> None:
            if _sha256(init_checkpoint) != init_checksum:
                raise RuntimeError("C0 checkpoint changed during silver adaptation")
            validation = dict(config.get("validation") or {})
            metrics = silver_validation_metrics(
                validation_rows or [],
                targets,
                binary_threshold=float(validation.get("binary_threshold", 0.5)),
                minimum_samples=int(validation.get("minimum_samples", 2)),
            )
            unwrapped = model.module if hasattr(model, "module") else model
            result = compact_result(
                config,
                status="completed",
                data={
                    "split_patients": (preflight.manifest or {}).get("split_patients", {}),
                    "split_studies": (preflight.manifest or {}).get("split_studies", {}),
                    "silver_statistics": statistics,
                },
                model={
                    **trainable_parameter_summary(unwrapped),
                    "encoder_parameters": encoder_parameter_report,
                    "peft": peft_report,
                    "auxiliary_heads": list(targets),
                    "auxiliary_heads_required_downstream": False,
                    "transfer": transfer_report,
                },
                compute={
                    "strategy": config["compute"]["strategy"],
                    "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                    "precision": config["compute"].get("precision"),
                    "training_time_min": training_result["training_time_min"],
                },
                evaluation={
                    "selection_metric": "negative_overall_masked_validation_loss",
                    "best_validation_metric": training_result["best_validation_metric"],
                    "per_target_validation": metrics,
                    "test_used": False,
                },
                reproducibility=reproducibility,
            )
            init_encoder_label = str((config.get("lineage") or {}).get("initialization", "C0"))
            result["silver_adaptation"] = {
                "stage": "silver_encoder_adaptation",
                "init_encoder": init_encoder_label,
                "init_checkpoint": str(init_checkpoint),
                "init_checkpoint_sha256": init_checksum,
                "init_experiment": checkpoint_init_experiment,
                "silver_source": config["silver_training"]["silver_source"],
                "silver_supervision": True,
                "eligible_targets": statistics["eligible"],
                "accepted_targets": statistics["accepted"],
                "abstained_targets": statistics["abstained"],
                "no_result_targets": statistics["no_result"],
                "effective_coverage": statistics["effective_coverage"],
                "targets_available": statistics["available_targets"],
                "encoder_output": "C_silver",
                "source_encoder_unchanged": True,
            }
            manager.write_result(run_dir, result)
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "init_encoder": init_encoder_label,
                        "silver_source": config["silver_training"]["silver_source"],
                        "encoder_output": "C_silver",
                        "output": str(run_dir),
                        "test_used": False,
                    },
                    indent=2,
                )
            )

        rank_zero_call(context, finalize)
        context.barrier()
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
