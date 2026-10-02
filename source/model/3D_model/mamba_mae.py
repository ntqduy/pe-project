"""3D Vision Mamba encoder of Mamba-MAE (third_party/repos/mamba_mae), MAE-pretrained.

The released checkpoint ``third_party/weights/Mamba_MAE.pth`` is ``mae_3d_vim_small_patch4``
(embed 384, 12 bidirectional-v2 Mamba blocks, patch 4^3) pre-trained on 160^3 four-contrast
BraTS MRI. The finetune architecture upstream pairs with it is
``vim_3D_small_patch4_stride4_224_bimambav2_final_pool_mean_abs_pos_embed_div2``; it is built
here with the same arguments except ``img_size`` (128), ``channels`` (1) and
``final_pool_type='all'`` (all tokens, so the token grid is available for Grad-CAM; the
embedding is still their mean, as upstream's 'mean' pooling). Two tensors are adapted, and
the log says so: the 4-contrast patch kernel is summed to one channel, and the 40^3 position
table is resized to 32^3. Upstream z-scores its MRI inputs, hence ``zscore`` intensity.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from source.model.base import (
    BaselineEncoder,
    intensity_from_config,
    load_state_with_report,
    resolve_pretrained,
    unwrap_state_dict,
)
from source.model.inflate import collapse_input_channels, interpolate_positions_3d
from source.model.weights import resolve_weight_path

from ._thirdparty import import_from_repo, repo_path, require_mamba_ssm

DEFAULT_WEIGHT = "third_party/weights/Mamba_MAE.pth"
PRETRAIN_INPUT = 160


class MambaMAECore(nn.Module):
    def __init__(self, input_size: int = 128, patch_size: int = 4, drop_path_rate: float = 0.1):
        super().__init__()
        require_mamba_ssm()
        repo_path("mamba_mae")
        models_vim = import_from_repo("mamba_mae", "models_vim", isolate=("rope", "models_vim"))
        self.net = models_vim.VisionMamba(
            img_size=int(input_size), patch_size=int(patch_size), stride=int(patch_size), embed_dim=384, depth=12,
            channels=1, num_classes=0, drop_path_rate=float(drop_path_rate),
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type="all",
            if_abs_pos_embed=True, if_rope=True, if_rope_residual=True, bimamba_type="v2",
            if_cls_token=False, if_devide_out=True, use_middle_cls_token=False, if_3d=True,
        )
        self.grid = (int(input_size) // int(patch_size),) * 3
        self.embed_dim = 384

    def forward(self, volume: Tensor) -> dict[str, Any]:
        tokens = self.net.forward_features(volume)                           # [B, L, C]
        feature_map = tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, *self.grid)
        # Pooled from feature_map so the Grad-CAM target is on the gradient path.
        return {"feature_map": feature_map, "global_embedding": feature_map.float().mean(dim=(2, 3, 4)),
                "pooling": "token_mean"}


def build_mamba_mae_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    input_size = int(config.get("input_size", 128))
    core = MambaMAECore(input_size=input_size, drop_path_rate=float(config.get("drop_path_rate", 0.1)))
    encoder = BaselineEncoder(
        core,
        core.embed_dim,
        "Mamba-MAE ViM-S 3D",
        intensity=intensity_from_config(config.get("intensity"), default_mode="zscore"),
        architecture_note=f"Model 3D: Vision Mamba 3D (patch 4^3, lưới token {core.grid}); Grad-CAM trên token map cuối",
        # The Mamba fast path reads in_proj.weight / out_proj.weight instead of calling them;
        # LoRALinear.weight is the merged W + BA, so the update is applied (and trained) there too.
        lora_target_modules=tuple(config.get("lora_target_modules") or ("mixer.in_proj", "mixer.out_proj")),
    )

    def load(options: Mapping[str, Any]) -> dict[str, Any]:
        path = resolve_weight_path(options.get("path") or DEFAULT_WEIGHT)
        if not path.is_file():
            raise FileNotFoundError(f"Mamba-MAE weight not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = unwrap_state_dict(payload)
        # Encoder only: the MAE decoder, mask token and (unused) cls token are dropped.
        state = {key: value for key, value in state.items()
                 if not key.startswith(("decoder", "mask_token", "cls_token"))}
        notes = []
        own = core.net.state_dict()
        kernel = state.get("patch_embed.proj.weight")
        if kernel is not None and kernel.shape[1] != own["patch_embed.proj.weight"].shape[1]:
            state["patch_embed.proj.weight"] = collapse_input_channels(kernel, 1)
            notes.append(f"patch kernel {tuple(kernel.shape)} -> 1 input channel (4 MRI contrasts summed)")
        table = state.get("pos_embed")
        if table is not None and tuple(table.shape) != tuple(own["pos_embed"].shape):
            side = round(table.shape[1] ** (1 / 3))
            state["pos_embed"] = interpolate_positions_3d(table, (side,) * 3, core.grid)
            notes.append(f"position table {side}^3 -> {core.grid} (trilinear)")
        # RoPE frequencies are a deterministic function of the token grid; keep the rebuilt ones.
        state = {key: value for key, value in state.items() if not key.startswith("rope.")}
        return load_state_with_report(core.net, state, source=str(path), notes=notes)

    return resolve_pretrained(encoder, config, load)
