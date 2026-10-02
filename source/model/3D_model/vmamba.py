"""VMamba 3D: the VMamba-v2 (VSSM) architecture with a 3-D cross-scan, ImageNet-inflated.

There is no public volumetric VMamba. This module rebuilds VMamba-B
(``vmambav2_base_224``: dims 128-1024, depths 2/2/15/2, d_state 1, ssm_ratio 2,
forward type ``v05_noz``, patch-embed v2, downsample v3) with 3-D operators and loads
``third_party/weights/vssm_base_VMamba.pth`` onto it:

* SS2D -> SS3D. VMamba's cross-scan visits a 2-D map [H, W] in K = 4 orders (row-major
  with W fastest, column-major with H fastest, and both reversed). The inflated 2-D kernels
  lie in the axial plane with H -> x and W -> y (source/model/inflate.py), so SS3D keeps
  K = 4 with the same in-plane orders, visiting the axial planes one after another (z
  slowest): flatten (z, x, y) (y fastest, = 2-D direction 0), flatten (z, y, x) (x fastest,
  = direction 1), and both reversed. Every per-direction SSM parameter (x_proj, dt_projs,
  A_logs, Ds) therefore keeps its shape, loads unchanged and drives the in-plane scan
  order it was pretrained on.
* Convolutions (patch embedding, depthwise 3x3, downsampling) are inflated to 3x3x3 (2-D
  kernel in the axial x-y plane, repeated along z); the RGB stem is summed to one channel.
  LayerNorms and MLPs are copied.

The selective scan itself is VMamba's own ``selective_scan_fn`` (CUDA kernel from mamba_ssm,
pure-PyTorch fallback on CPU), imported from third_party/repos/VMamba unmodified.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from source.model.base import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BaselineEncoder,
    intensity_from_config,
    load_state_with_report,
    resolve_pretrained,
    unwrap_state_dict,
)
from source.model.inflate import inflate_state_dict
from source.model.weights import resolve_weight_path

from ._thirdparty import import_from_repo

DEFAULT_WEIGHT = "third_party/weights/vssm_base_VMamba.pth"
VARIANTS = {"base": ((2, 2, 15, 2), 128)}


def _vmamba():
    return import_from_repo("VMamba", "vmamba")


class Permute(nn.Module):
    def __init__(self, *order: int):
        super().__init__()
        self.order = order

    def forward(self, values: Tensor) -> Tensor:
        return values.permute(*self.order)


def cross_scan_3d(x: Tensor) -> Tensor:
    """[B, C, x, y, z] -> [B, 4, C, L]: (z,x,y), (z,y,x) and both reversed.

    Within each axial plane these are VMamba's 2-D orders on [H, W] = [x, y]: direction 0 is
    row-major (y fastest), direction 1 column-major (x fastest); z is always the slowest axis.
    """
    row_major = x.permute(0, 1, 4, 2, 3).flatten(2)          # [B, C, z, x, y]
    column_major = x.permute(0, 1, 4, 3, 2).flatten(2)       # [B, C, z, y, x]
    return torch.stack([row_major, column_major, row_major.flip(-1), column_major.flip(-1)], dim=1)


def cross_merge_3d(y: Tensor, shape: Sequence[int]) -> Tensor:
    """Inverse of :func:`cross_scan_3d`, summing the four directions -> [B, C, x, y, z]."""
    batch, _, channels, _ = y.shape
    x_size, y_size, z_size = shape
    row_major = (y[:, 0] + y[:, 2].flip(-1)).view(batch, channels, z_size, x_size, y_size)
    column_major = (y[:, 1] + y[:, 3].flip(-1)).view(batch, channels, z_size, y_size, x_size)
    return row_major.permute(0, 1, 3, 4, 2) + column_major.permute(0, 1, 4, 3, 2)


class SS3D(nn.Module):
    def __init__(self, d_model: int, d_state: int = 1, ssm_ratio: float = 2.0, d_conv: int = 3, k_group: int = 4):
        super().__init__()
        vm = _vmamba()
        self.selective_scan_fn = vm.selective_scan_fn
        self.d_inner = int(ssm_ratio * d_model)
        self.d_state = int(d_state)
        self.dt_rank = math.ceil(d_model / 16)
        self.k_group = int(k_group)
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.conv3d = nn.Conv3d(self.d_inner, self.d_inner, d_conv, padding=(d_conv - 1) // 2, groups=self.d_inner, bias=False)
        self.act = nn.SiLU()
        self.x_proj_weight = nn.Parameter(
            torch.randn(self.k_group, self.dt_rank + 2 * self.d_state, self.d_inner) * self.d_inner ** -0.5
        )
        A_logs, Ds, dt_weight, dt_bias = vm.mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, 1.0, "random", 0.001, 0.1, 1e-4, k_group=self.k_group
        )
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = A_logs, Ds, dt_weight, dt_bias
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:                    # channel-last [B, D, H, W, C]
        x = self.in_proj(x).permute(0, 4, 1, 2, 3)
        x = self.act(self.conv3d(x))
        batch, channels, *shape = x.shape
        length = shape[0] * shape[1] * shape[2]
        # The selective-scan CUDA kernel needs u, delta, B, C in one dtype; under bf16 autocast
        # the einsums would return bf16 next to an fp32 u. Run the whole SSM core in fp32.
        with torch.autocast(x.device.type, enabled=False):
            xs = cross_scan_3d(x.float())                      # [B, K, C, L]
            x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight.float())
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight.float())
            ys = self.selective_scan_fn(
                xs.reshape(batch, -1, length).contiguous(), dts.reshape(batch, -1, length).contiguous(),
                -torch.exp(self.A_logs.float()), Bs.float().contiguous(), Cs.float().contiguous(), self.Ds.float(),
                self.dt_projs_bias.reshape(-1).float(), True, True,
            ).float().view(batch, self.k_group, channels, length)
        y = cross_merge_3d(ys, shape).permute(0, 2, 3, 4, 1)   # channel-last, fp32
        return self.out_proj(self.out_norm(y))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, values: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(values)))


class VSSBlock3D(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.op = SS3D(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = float(drop_path)

    def _drop(self, values: Tensor) -> Tensor:
        if not self.training or self.drop_path <= 0:
            return values
        keep = torch.rand(values.shape[0], *([1] * (values.ndim - 1)), device=values.device) >= self.drop_path
        return values * keep.to(values.dtype) / (1 - self.drop_path)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self._drop(self.op(self.norm(values)))
        return values + self._drop(self.mlp(self.norm2(values)))


class VSSStage3D(nn.Module):
    def __init__(self, dim: int, depth: int, drop_paths: Sequence[float], downsample: bool):
        super().__init__()
        self.blocks = nn.Sequential(*(VSSBlock3D(dim, drop_paths[index]) for index in range(depth)))
        self.downsample = (
            nn.Sequential(Permute(0, 4, 1, 2, 3), nn.Conv3d(dim, 2 * dim, 3, stride=2, padding=1),
                          Permute(0, 2, 3, 4, 1), nn.LayerNorm(2 * dim))
            if downsample else nn.Identity()
        )


class VMamba3dCore(nn.Module):
    def __init__(self, depths: Sequence[int], embed_dim: int, drop_path_rate: float = 0.2, gradient_checkpointing: bool = True):
        super().__init__()
        half = embed_dim // 2
        self.patch_embed = nn.Sequential(
            nn.Conv3d(1, half, 3, stride=2, padding=1), Permute(0, 2, 3, 4, 1), nn.LayerNorm(half), Permute(0, 4, 1, 2, 3),
            nn.GELU(), nn.Conv3d(half, embed_dim, 3, stride=2, padding=1), Permute(0, 2, 3, 4, 1), nn.LayerNorm(embed_dim),
        )
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        dims = [embed_dim * 2 ** index for index in range(len(depths))]
        stages, cursor = [], 0
        for index, depth in enumerate(depths):
            stages.append(VSSStage3D(dims[index], depth, rates[cursor : cursor + depth], downsample=index < len(depths) - 1))
            cursor += depth
        self.layers = nn.ModuleList(stages)
        self.out_dim = dims[-1]
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, volume: Tensor) -> dict[str, Any]:
        values = self.patch_embed(volume)                      # channel-last
        pyramid = []
        for stage in self.layers:
            for block in stage.blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    values = checkpoint(block, values, use_reentrant=False)
                else:
                    values = block(values)
            pyramid.append(values.permute(0, 4, 1, 2, 3))
            values = stage.downsample(values)
        return {"feature_map": pyramid[-1], "pyramid": pyramid}


def _rename(key: str) -> str | None:
    if key.startswith("classifier"):
        return None
    return key.replace(".op.conv2d.", ".op.conv3d.")


def build_vmamba_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    variant = str(config.get("variant") or "base")
    depths, embed_dim = VARIANTS[variant]
    depths = tuple(config.get("depths") or depths)
    core = VMamba3dCore(depths, embed_dim, drop_path_rate=float(config.get("drop_path_rate", 0.2)),
                        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)))
    default_intensity = {"mode": "window", "window": [-250.0, 450.0], "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
    encoder = BaselineEncoder(
        core,
        core.out_dim,
        f"VMamba-{variant[0].upper()} 3D",
        intensity=intensity_from_config(config.get("intensity") or default_intensity),
        architecture_note="Model 3D: VMamba 3D (SS3D, 4 hướng quét 3D), một forward trên cả volume; feature map = stage 4",
        lora_target_modules=tuple(config.get("lora_target_modules") or ("op.in_proj", "op.out_proj")),
    )

    def load(options: Mapping[str, Any]) -> dict[str, Any]:
        path = resolve_weight_path(options.get("path") or DEFAULT_WEIGHT)
        if not path.is_file():
            raise FileNotFoundError(f"VMamba weight not found: {path}")
        source = unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        state, summary = inflate_state_dict(source, core.state_dict(), rename=_rename)
        notes = [f"ImageNet VMamba-B 2D -> 3D (copied {summary['copied']}, inflated {summary['inflated']})",
                 "SS2D 4-direction parameters reused for the 4 SS3D scan orders", "RGB stem summed to 1 channel"]
        return load_state_with_report(core, state, source=str(path), notes=notes)

    return resolve_pretrained(encoder, config, load)
