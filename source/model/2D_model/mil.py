"""Slice multiple-instance learning: 2-D backbones used over a whole CT volume.

``2d`` uses N uniformly spaced axial slices. ``2.5d`` uses the same N centres, with
the adjacent axial slices as the three input channels. Gated attention pools their feature
vectors into one study embedding. The returned map is [B, C, h, w, N] for Grad-CAM.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class GatedAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.value, self.gate, self.score = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)

    def forward(self, instances: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.score(torch.tanh(self.value(instances)) * torch.sigmoid(self.gate(instances))).squeeze(-1)
        weights = torch.softmax(logits.float(), dim=1).to(instances.dtype)
        return (weights.unsqueeze(-1) * instances).sum(dim=1), weights


def uniform_slice_indices(depth: int, count: int) -> list[int]:
    count = max(1, min(int(count), int(depth)))
    return [min(depth - 1, int((index + 0.5) * depth / count)) for index in range(count)]


class SliceMILCore(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, *, mode: str = "2d", num_slices: int = 32,
                 slice_size: int = 224, slice_offset: int = 1, pooling: str = "attention",
                 attention_hidden: int = 128, chunk_size: int = 32, gradient_checkpointing: bool = True,
                 normalize_mean: Sequence[float] | None = None, normalize_std: Sequence[float] | None = None):
        super().__init__()
        self.backbone, self.feature_dim = backbone, int(feature_dim)
        self.mode = str(mode).lower().replace(".", "")
        if self.mode not in {"2d", "25d"}:
            raise ValueError(f"slice mode must be 2d or 2.5d, got {mode!r}")
        self.num_slices, self.slice_size, self.slice_offset = int(num_slices), int(slice_size), int(slice_offset)
        self.pooling = str(pooling).lower()
        if self.pooling not in {"attention", "mean", "max"}:
            raise ValueError(f"MIL pooling must be attention, mean or max, got {pooling!r}")
        self.attention = GatedAttentionPool(self.feature_dim, int(attention_hidden)) if self.pooling == "attention" else None
        self.chunk_size, self.gradient_checkpointing = max(1, int(chunk_size)), bool(gradient_checkpointing)
        channels = 3 if self.mode == "25d" else 1
        if normalize_mean is not None and normalize_std is not None:
            mean, std = torch.as_tensor(list(normalize_mean), dtype=torch.float32), torch.as_tensor(list(normalize_std), dtype=torch.float32)
            if channels == 1:
                mean, std = mean.mean().reshape(1), std.mean().reshape(1)
            self.register_buffer("norm_mean", mean.reshape(1, channels, 1, 1), persistent=False)
            self.register_buffer("norm_std", std.reshape(1, channels, 1, 1), persistent=False)
        else:
            self.norm_mean = self.norm_std = None

    def channel_slices(self, indices: Sequence[int], depth: int) -> list[list[int]]:
        """Axial slices behind each instance: [i] in 2D, [i - offset, i, i + offset] in 2.5D."""
        if self.mode == "2d":
            return [[int(index)] for index in indices]
        offsets = (-self.slice_offset, 0, self.slice_offset)
        return [[min(depth - 1, max(0, int(index) + offset)) for offset in offsets] for index in indices]

    def _instances(self, volume: Tensor, indices: Sequence[int]) -> Tensor:
        batch, _, x_size, y_size, depth = volume.shape
        if self.mode == "2d":
            stack = volume[..., list(indices)]
        else:
            groups = self.channel_slices(indices, depth)
            channels = [volume[:, 0, :, :, [group[position] for group in groups]] for position in range(3)]
            stack = torch.stack(channels, dim=1)
        images = stack.permute(0, 4, 1, 2, 3).reshape(batch * len(indices), stack.shape[1], x_size, y_size)
        if (x_size, y_size) != (self.slice_size, self.slice_size):
            images = F.interpolate(images, size=(self.slice_size, self.slice_size), mode="bilinear", align_corners=False)
        if self.norm_mean is not None:
            images = (images - self.norm_mean.to(images.dtype)) / self.norm_std.to(images.dtype)
        return images

    def _encode(self, images: Tensor) -> Tensor:
        outputs = []
        for start in range(0, images.shape[0], self.chunk_size):
            chunk = images[start:start + self.chunk_size]
            outputs.append(checkpoint(self.backbone, chunk, use_reentrant=False)
                           if self.gradient_checkpointing and self.training and torch.is_grad_enabled() else self.backbone(chunk))
        return torch.cat(outputs, dim=0)

    def forward(self, volume: Tensor) -> dict[str, object]:
        if volume.ndim != 5:
            raise ValueError(f"slice MIL expects [B, C, x, y, z], got {tuple(volume.shape)}")
        batch, depth = volume.shape[0], volume.shape[-1]
        indices = uniform_slice_indices(depth, self.num_slices)
        maps = self._encode(self._instances(volume, indices))
        maps = maps.reshape(batch, len(indices), *maps.shape[1:])
        feature_map = maps.permute(0, 2, 3, 4, 1)
        instances = feature_map.float().mean(dim=(2, 3)).transpose(1, 2)
        count = len(indices)
        if self.pooling == "attention":
            pooled, weights = self.attention(instances)
        elif self.pooling == "mean":
            pooled = instances.mean(dim=1)
            weights = torch.full(instances.shape[:2], 1.0 / count, device=instances.device)
        else:
            pooled, argmax = instances.max(dim=1)
            weights = F.one_hot(argmax, count).float().mean(dim=1)
        return {"feature_map": feature_map, "global_embedding": pooled, "pooling": f"slice_mil_{self.pooling}",
                "metadata": {"slice_indices": list(indices), "slice_attention": weights.detach(),
                             "slice_mode": "2.5d" if self.mode == "25d" else "2d",
                             # ROI pooling reads region masks on exactly these slices.
                             "slice_channel_indices": self.channel_slices(indices, depth),
                             "slice_depth": int(depth)}}


__all__ = ["GatedAttentionPool", "SliceMILCore", "uniform_slice_indices"]
