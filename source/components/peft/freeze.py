from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from .lora import inject_lora


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def apply_peft(module: nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    method = str(config.get("method", "full")).lower()
    if method in {"frozen", "linear_probe"}:
        set_requires_grad(module, False)
        return {"method": method, "modified_modules": []}
    if method == "full":
        set_requires_grad(module, True)
        return {"method": method, "modified_modules": []}
    if method == "lora":
        set_requires_grad(module, False)
        # Layer names are backbone specific: a transformer has attn/projection Linears, the
        # CT-FM SegResNet encoder only convolutions. A backbone that declares its own LoRA
        # targets (backbones.yaml lora_target_modules) therefore takes precedence over the
        # generic peft.target_modules.
        backbone_targets = tuple(getattr(module, "lora_target_modules", ()) or ())
        targets = backbone_targets or tuple(config.get("target_modules") or ())
        replaced = inject_lora(
            module,
            targets,
            rank=int(config.get("rank", 8)),
            alpha=float(config.get("alpha", 16)),
            dropout=float(config.get("dropout", 0)),
        )
        return {
            "method": method,
            "modified_modules": replaced,
            "target_modules": list(targets),
            "target_source": "backbone" if backbone_targets else "peft.target_modules",
        }
    raise ValueError(f"unsupported PEFT method: {method}")


def trainable_parameter_summary(module: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    lora = sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and ("lora_a" in name or "lora_b" in name)
    )
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
        "lora_params": lora,
    }
