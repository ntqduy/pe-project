from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    # Some upstream code reads ``layer.weight`` / ``layer.bias`` instead of calling the layer
    # (the Mamba fast path: ``self.in_proj.weight @ x``, ``F.linear(y, self.out_proj.weight)``).
    # ``weight`` is therefore the merged ``W + scale * B A``, so that path still applies and
    # trains the LoRA update (LoRA dropout acts only through forward()).
    @property
    def weight(self) -> Tensor:
        return self.base.weight + (self.lora_b @ self.lora_a) * self.scale

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, values: Tensor) -> Tensor:
        update = (self.dropout(values) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(values) + update * self.scale


class LoRAConv(nn.Module):
    """LoRA for convolutions: a rank-r conv with the base kernel, then a 1x1 conv back.

    The update ``B(A(x))`` has the same receptive field, stride and padding as the frozen
    base convolution; B starts at zero, so the wrapped layer initially equals the base.
    """

    def __init__(self, base: nn.Conv1d | nn.Conv2d | nn.Conv3d, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if base.groups != 1:
            raise ValueError("LoRA does not support grouped convolutions")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        conv = type(base)
        self.lora_a = conv(
            base.in_channels, rank, base.kernel_size, stride=base.stride, padding=base.padding,
            dilation=base.dilation, bias=False, padding_mode=base.padding_mode,
        )
        self.lora_b = conv(rank, base.out_channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self) -> Tensor:
        """Merged kernel: a 1x1 conv after A equals one conv with kernel ``B @ A`` (see LoRALinear)."""
        update = torch.einsum("or,r...->o...", self.lora_b.weight.flatten(1), self.lora_a.weight)
        return self.base.weight + update * self.scale

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    # Read-only geometry of the frozen base conv, for upstream code that inspects a layer
    # (``conv.stride``, ``conv.out_channels``, ...) instead of only calling it.
    @property
    def in_channels(self) -> int:
        return self.base.in_channels

    @property
    def out_channels(self) -> int:
        return self.base.out_channels

    @property
    def kernel_size(self) -> tuple[int, ...]:
        return self.base.kernel_size

    @property
    def stride(self) -> tuple[int, ...]:
        return self.base.stride

    @property
    def padding(self) -> tuple[int, ...] | str:
        return self.base.padding

    @property
    def dilation(self) -> tuple[int, ...]:
        return self.base.dilation

    @property
    def groups(self) -> int:
        return self.base.groups

    @property
    def padding_mode(self) -> str:
        return self.base.padding_mode

    def forward(self, values: Tensor) -> Tensor:
        return self.base(values) + self.lora_b(self.lora_a(self.dropout(values))) * self.scale


_LORA_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def inject_lora(
    module: nn.Module,
    target_modules: Iterable[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Wrap every Linear/Conv layer whose qualified name contains a target token.

    Grouped / depthwise convolutions (ConvNeXt ``conv_dw``, PENet's ResNeXt ``conv2``, Mamba's
    ``conv1d``) have no low-rank update of this form and are skipped, so a stage-level token
    wraps that stage's pointwise convs and Linears; only a target list that matches no
    eligible layer at all is an error.
    """
    targets = tuple(target_modules)
    if not targets:
        raise ValueError("LoRA target_modules cannot be empty")
    replaced: list[str] = []
    for full_name, child in list(module.named_modules()):
        if not isinstance(child, _LORA_TYPES) or not any(token in full_name for token in targets):
            continue
        if ".lora_a" in full_name or ".lora_b" in full_name:
            continue
        if not isinstance(child, nn.Linear) and child.groups != 1:
            continue
        parent_name, _, attribute = full_name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        wrapped = (
            LoRALinear(child, rank, alpha, dropout)
            if isinstance(child, nn.Linear)
            else LoRAConv(child, rank, alpha, dropout)
        )
        setattr(parent, attribute, wrapped)
        replaced.append(full_name)
    if not replaced:
        raise ValueError(f"no Linear/ungrouped Conv modules matched LoRA targets: {targets}")
    return replaced
