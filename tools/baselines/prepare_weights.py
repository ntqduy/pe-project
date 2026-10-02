#!/usr/bin/env python
"""Fetch every baseline's pretrained weights once and report what loads.

    python tools/baselines/prepare_weights.py                 # all 20 arms
    python tools/baselines/prepare_weights.py --models vit_3d swin_3d

Each arm's encoder is built exactly as training builds it (its run config + backbones.yaml),
on CPU, which downloads what is missing (timm / MedicalNet into the Hugging Face cache,
Swin-UNETR SSL into third_party/weights/baselines/) so parallel training jobs never race on a
download. The table (stdout + <outputs>/diagnosis/BASE/weights_status.md) lists per arm:
dimension, weight source, LOADED / SCRATCH / UNAVAILABLE and matched/total tensors.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.baselines.experiments import EXPERIMENTS, config_path, model_dimension, outputs_root  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=EXPERIMENTS["exp01_baselines"]["models"])
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    from source.engine.factory import build_task_model
    from source.utils.config import load_config, validate_config

    rows = []
    for model in args.models:
        try:
            config = validate_config({k: v for k, v in load_config(config_path(model), ["compute.devices=[]"]).items()
                                      if k != "config_hash"})
            built, _ = build_task_model(config)
            report = dict(getattr(built.image_encoder, "pretrained_report", {}) or {})
            status = {"loaded": "LOADED", "scratch": "SCRATCH", "delegated": "LOADED (cached features)"}.get(
                str(report.get("status")), str(report.get("status")))
            matched = (f"{report.get('matched_tensors')}/{report.get('model_tensors')}"
                       if report.get("matched_tensors") is not None else "-")
            detail = report.get("reason") or "; ".join(report.get("notes") or [])
            source = report.get("source") or "-"
            params = sum(parameter.numel() for parameter in built.image_encoder.parameters()) / 1e6
        except Exception as exc:  # noqa: BLE001 - one missing dependency must not hide the others
            traceback.print_exc(limit=1)
            status, matched, source, params = "UNAVAILABLE", "-", "-", float("nan")
            detail = f"{type(exc).__name__}: {str(exc)[:200]}"
        rows.append((model, model_dimension(model), status, matched, f"{params:.1f}", str(source), str(detail)))
        print(f"{model:16s} {rows[-1][1]:5s} {status:24s} {matched:9s} {source}", flush=True)

    lines = ["| model | dim | weights | matched tensors | encoder params (M) | source | notes |",
             "|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(value.replace("|", "/") for value in row) + " |" for row in rows]
    table = "\n".join(lines)
    print("\n" + table)
    if not args.no_write:
        destination = outputs_root() / "diagnosis" / "BASE" / "weights_status.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("# Baseline pretrained-weight status\n\n" + table + "\n", encoding="utf-8")
        print(f"\nwritten: {destination}")
    return 0 if all(row[2] != "UNAVAILABLE" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
