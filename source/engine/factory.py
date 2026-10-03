from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

# CT-FM's timm dependency still registers a legacy TorchScript interface. It is
# unrelated to this experiment and does not indicate a model/runtime failure.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.interface` is deprecated\. Please use `torch\.compile` instead\.",
)

from torch import nn

from source.components.encoders.image.registry import build_image_encoder
from source.components.peft.freeze import apply_peft


# task.architecture of the 2D / 2.5D / 3D baseline zoo (source/model): encoder -> shared
# projection -> MLP or KAN head, for diagnosis and image-only prognosis alike.
BASELINE_ARCHITECTURE = "baseline_classifier"


def _build_baseline_classifier(
    config: Mapping[str, Any], stage: str, task: Mapping[str, Any], model_config: Mapping[str, Any]
) -> tuple[nn.Module, dict[str, Any]]:
    from source.model.classifier import BaselineClassifier

    if stage not in {"diagnosis", "prognosis"}:
        raise ValueError(f"{BASELINE_ARCHITECTURE} supports diagnosis and prognosis, got stage={stage!r}")
    if stage == "prognosis" and set(task.get("modalities") or ("image",)) != {"image"}:
        raise ValueError(f"{BASELINE_ARCHITECTURE} is image-only; set task.modalities: [image]")
    primary = str(task.get("primary_target") or ("pe_present" if stage == "diagnosis" else "mortality_30d"))
    targets = task.get("targets") or {primary: 1}
    if isinstance(targets, list):
        targets = {name: 1 for name in targets}
    wants_weights = bool((model_config.get("pretrained") or {}).get("enabled", True))
    external = str(model_config.get("integration") or "external") != "baseline" and "factory" in model_config
    if external and not wants_weights:
        # --scratch on a CT-FM arm: model.load_pretrained is forced on by encoder.init_source,
        # so the baseline switch model.pretrained.enabled is honoured here.
        if bool(model_config.get("cached_features")):
            raise ValueError("cached CT-FM features were extracted with pretrained weights; a scratch run is impossible")
        model_config = {**model_config, "load_pretrained": False}
    encoder = build_image_encoder(model_config)
    if not hasattr(encoder, "pretrained_report"):
        # CT-FM encoders predate the baseline zoo; report what their own loader did.
        from source.model.base import format_pretrained_report, scratch_report

        load = getattr(encoder, "load_report", None)
        if bool(model_config.get("cached_features")):
            encoder.pretrained_report = {
                "status": "delegated", "source": str(model_config.get("checkpoint")),
                "reason": "cached features were extracted with these frozen weights (tools/data/build_ctfm_cache.py)",
            }
        elif load:
            missing, unexpected = list(load.get("missing_keys") or []), list(load.get("unexpected_keys") or [])
            total = int(load.get("model_tensors") or 0)
            encoder.pretrained_report = {
                "status": "loaded", "source": load.get("checkpoint"),
                "matched_tensors": total - len(missing), "model_tensors": total,
                "missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatch": [],
                "notes": ["strict load" if load.get("strict") else "non-strict load"],
            }
        else:
            encoder.pretrained_report = scratch_report("model.load_pretrained=false / --scratch")
        print(format_pretrained_report(str(getattr(encoder, "backbone_name", model_config.get("backbone"))),
                                       encoder.pretrained_report), flush=True)
    peft_report = apply_peft(encoder, dict(config.get("peft") or {"method": "full"}))
    model = BaselineClassifier(
        encoder, stage, targets, primary_target=primary, head=dict(config.get("head") or {})
    )
    return model, peft_report


def build_task_model(config: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    """The task model of a run: every trainable arm is a baseline classifier (source/model).

    Encoder (2D / 2.5D slice-MIL, 3D, CT-FM LoRA or cached CT-FM features) -> shared projection
    -> MLP or KAN head, for diagnosis and image-only prognosis.
    """
    experiment = dict(config.get("experiment") or {})
    stage = str(experiment.get("stage"))
    model_config = dict(config.get("model") or {})
    model_config["data_mode"] = str((config.get("data") or {}).get("mode"))
    task = dict(config.get("task") or {})
    architecture = str(task.get("architecture") or "")
    if architecture != BASELINE_ARCHITECTURE:
        raise ValueError(
            f"task.architecture={architecture!r} is not supported; every trainable run uses "
            f"{BASELINE_ARCHITECTURE!r} (configs/components/baselines.yaml)"
        )
    return _build_baseline_classifier(config, stage, task, model_config)
