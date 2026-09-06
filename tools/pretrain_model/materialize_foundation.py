from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from source.components.encoders.image.registry import build_image_encoder
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.engine.checkpoint import save_checkpoint_atomic
from source.engine.experiment import OutputManager, compact_result
from source.pretraining.foundation import FoundationModel
from source.profiling.model_profile import profile_model
from source.utils.environment import environment_report
from source.utils.logger import RunLogger

from tools._common import base_parser, build_dataset, build_training_lineage, resolve_cli_config


def main() -> int:
    parser = base_parser("Materialize and profile a pinned public CT foundation encoder")
    args = parser.parse_args()
    if args.resume:
        raise SystemExit("--resume is not applicable to foundation materialization")
    config = resolve_cli_config(args)
    paths = ProjectPaths.resolve(config)
    preflight = require_preflight(config, paths)
    manager = OutputManager(paths)
    experiment_id = str(config["experiment"]["id"])
    run_dir = manager.prepare("foundation", experiment_id, resume=args.resume, overwrite=args.overwrite)
    manager.write_config(run_dir, {**config, "resolved_paths": paths.as_dict()})
    encoder = build_image_encoder({**dict(config["model"]), "data_mode": str(config["data"]["mode"])})
    model = FoundationModel(encoder)
    dataset = build_dataset(config, paths, "train")
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    use_cuda = torch.cuda.is_available() and bool(config["compute"].get("devices"))
    device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
    model.to(device)
    volume = batch["volume"].to(device)
    profile = profile_model(
        model,
        lambda: model(volume),
        warmup=1,
        iterations=int((config.get("profiling") or {}).get("iterations", 5)),
    )
    environment = environment_report(paths.code_root)
    lineage = build_training_lineage(
        config,
        paths,
        code_commit=environment["git_commit"],
        source_checkpoint=config["model"]["checkpoint"],
    )
    save_checkpoint_atomic(run_dir / "best.ckpt", model, lineage=lineage)
    RunLogger(run_dir / "logs" / "train.log", echo=False).log(
        f"materialized backbone={config['model']['backbone']} "
        f"latency_ms={profile['latency_ms_per_volume']:.3f}"
    )
    manifest_audit = preflight.manifest or {}
    manager.write_result(
        run_dir,
        compact_result(
            config,
            status="completed",
            data={
                "profile_volume": str(dataset.rows[0]["study_id"]),
                "split_patients": manifest_audit.get("split_patients", {}),
            },
            model=profile,
            compute={
                "strategy": "single",
                "gpu_count": 1 if device.type == "cuda" else 0,
                "peak_vram_gb": profile["peak_vram_gb"],
                "latency_ms_per_volume": profile["latency_ms_per_volume"],
            },
            reproducibility=environment,
        ),
    )
    print(json.dumps({"status": "completed", "output": str(run_dir), "profile": profile}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
