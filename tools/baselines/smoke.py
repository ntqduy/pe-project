#!/usr/bin/env python
"""One real training step per baseline arm, at the real input size, before any long run.

    python tools/baselines/smoke.py                          # all arms, cuda:0, bf16, 128^3
    python tools/baselines/smoke.py --models vmamba_3d mamba_mae_3d --batch-size 1
    python tools/baselines/smoke.py --device cpu --size 32   # quick wiring check without a GPU

Per arm: build it exactly as training does (run config + backbones.yaml, real weights),
run forward + backward through the real diagnosis loss under the configured precision
(bf16 autocast on CUDA), and check that every trainable parameter got a gradient and that
the Grad-CAM target (the encoder's 5-D feature map) receives one. Reports seconds per step,
peak VRAM (i.e. whether the configured micro-batch fits), total / trainable parameters and
where the weights came from. CT-FM arms are also audited: frozen = nothing inside the
encoder trains; LoRA = only LoRA adapters inside the encoder train. --report writes the
table as markdown; nothing else is written.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.baselines.experiments import EXPERIMENTS, config_path, model_dimension  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=EXPERIMENTS["exp01_baselines"]["models"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=128, help="cube side of the synthetic volume")
    parser.add_argument("--batch-size", type=int, help="default: each arm's training.batch_size")
    parser.add_argument("--head", default="mlp", choices=["mlp", "kan"])
    parser.add_argument("--precision", default=None, help="auto | bf16 | fp16 | fp32 (default: compute.precision)")
    parser.add_argument("--set", dest="extra", action="append", default=[], metavar="KEY=VALUE",
                        help="config override for every arm (e.g. model.checkpoint=... for a local CT-FM copy)")
    parser.add_argument("--report", type=Path, help="also write the table as markdown here")
    args = parser.parse_args(argv)

    import torch

    from source.engine.factory import build_task_model
    from source.engine.task_steps import task_loss_step
    from source.utils.config import load_config, validate_config

    device = torch.device(args.device)
    rows, failed = [], 0
    for name in args.models:
        started = time.time()
        try:
            overrides = ["compute.devices=[]", f"head.type={args.head}", *args.extra]
            if name == "vit_3d":
                overrides.append(f"model.input_size={args.size}")
            if name == "mamba_mae_3d":
                overrides.append(f"model.input_size={args.size}")
            config = validate_config({key: value for key, value in load_config(
                config_path(name), overrides).items() if key != "config_hash"})
            from source.engine.trainer import resolve_precision

            precision = resolve_precision(args.precision or str(config["compute"].get("precision", "fp32")), device)
            batch = int(args.batch_size or config["training"]["batch_size"])
            model, _ = build_task_model(config)
            model.to(device).train()
            if name == "ctfm_frozen_3d":
                volume = torch.randn(batch, 513, 1, 1, 1, device=device)
                volume[:, -1] = 1.0
                model.fit_input_standardizer([{"volume": volume}, {"volume": volume + 0.5}], device)
                model.train()
                # The frozen encoder is a passthrough: as in the Grad-CAM preview, the cached
                # feature grid itself is the leaf the CAM gradient reaches.
                volume = volume.detach().requires_grad_(True)
            else:
                volume = torch.rand(batch, 1, args.size, args.size, args.size, device=device)
            labels = torch.zeros(batch, 1, device=device)
            labels[0] = 1.0
            data = {"volume": volume, "masks": {}, "labels": labels,
                    "label_valid": torch.ones(batch, 1, dtype=torch.bool, device=device),
                    "patient_id": [str(i) for i in range(batch)], "study_id": [str(i) for i in range(batch)]}
            state: dict = {}

            def hook(_module, _inputs, output):
                feature_map = output["feature_map"] if isinstance(output, dict) else getattr(output, "feature_map", None)
                if feature_map is None and isinstance(output, (list, tuple)):
                    feature_map = output[-1]
                if feature_map is not None and feature_map.requires_grad:
                    feature_map.retain_grad()
                state["feature_map"] = feature_map

            encoder = model.image_encoder
            target = encoder.model if isinstance(getattr(encoder, "model", None), torch.nn.Module) else encoder
            handle = target.register_forward_hook(hook)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            step = time.time()
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and precision in {"bf16", "fp16"}):
                loss, _ = task_loss_step(config)(model, data)
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.time() - step
            handle.remove()
            trainable = [p for p in model.parameters() if p.requires_grad]
            with_grad = sum(p.grad is not None for p in trainable)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in trainable)
            audit = trainable_audit(name, model)
            feature_map = state.get("feature_map")
            cam = "ok" if feature_map is not None and feature_map.grad is not None and float(feature_map.grad.abs().sum()) > 0 else "NO GRAD"
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else float("nan")
            report = getattr(encoder, "pretrained_report", {}) or {}
            weights = str(report.get("status"))
            source = report.get("source") or report.get("reason") or ""
            if report.get("matched_tensors") is not None:
                source = f"{source} (matched {report.get('matched_tensors')}/{report.get('model_tensors', '?')})"
            if not report and name.startswith("ctfm"):
                weights, source = "CT-FM", str(config["model"].get("checkpoint") or "")
            # PENet keeps upstream stochastic depth: skipped blocks get no gradient in a step.
            stochastic = bool(getattr(encoder, "ddp_find_unused_parameters", False))
            ok = ((with_grad == len(trainable) or (stochastic and with_grad > 0)) and cam == "ok"
                  and torch.isfinite(loss).item() and not audit.startswith("FAIL"))
            failed += 0 if ok else 1
            rows.append((name, model_dimension(name), "OK" if ok else "CHECK", f"{float(loss.detach()):.4f}", f"{with_grad}/{len(trainable)}",
                         cam, f"{seconds:.2f}", f"{peak:.1f}", str(batch), precision,
                         f"{total_params / 1e6:.2f}M / {trainable_params / 1e6:.3f}M", audit, f"{weights}: {source}"))
            del model, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - report every arm
            traceback.print_exc(limit=3)
            failed += 1
            rows.append((name, model_dimension(name), "FAIL", "-", "-", "-", "-", "-", "-", "-", "-", "-",
                         f"{type(exc).__name__}: {str(exc)[:160]}"))
        print(f"{rows[-1][0]:16s} {rows[-1][2]:5s} {time.time() - started:6.1f}s", flush=True)
    header = ("model", "dim", "status", "loss", "grads", "cam", "s/step", "peak GB", "batch", "precision",
              "params total / trainable", "trainable audit", "weights / error")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in rows]
    print("\n" + "\n".join(lines))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            f"# Baseline arm smoke (device={args.device}, size={args.size}^3, head={args.head})\n\n"
            + "\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


def trainable_audit(name: str, model) -> str:
    """What trains inside the encoder: nothing (frozen CT-FM), only LoRA (CT-FM LoRA), or all."""
    names = [parameter_name for parameter_name, parameter in model.named_parameters() if parameter.requires_grad]
    inside = [item for item in names if item.startswith("image_encoder.")]
    outside = len(names) - len(inside)
    lora = [item for item in inside if "lora" in item.lower()]
    other = [item for item in inside if "lora" not in item.lower()]
    if name == "ctfm_frozen_3d":
        return (f"PASS frozen: encoder 0 trainable, head/projection {outside}" if not inside
                else f"FAIL frozen encoder trains {len(inside)} tensors, e.g. {inside[:2]}")
    if name == "ctfm_lora_3d":
        return (f"PASS LoRA: encoder {len(lora)} LoRA tensors only, head/projection {outside}" if lora and not other
                else f"FAIL LoRA: {len(lora)} LoRA + {len(other)} other encoder tensors train, e.g. {other[:2]}")
    return f"full: encoder {len(inside)}, head/projection {outside}"


if __name__ == "__main__":
    raise SystemExit(main())
