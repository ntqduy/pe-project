"""timm 2-D backbones returning a channel-first spatial map for every architecture family."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn


class Timm2DBackbone(nn.Module):
    """``timm.create_model(..., num_classes=0)`` with ``forward -> [N, C, h, w]``.

    CNNs already return NCHW; ViTs return tokens (prefix tokens are dropped, the patch grid
    is restored); Swin returns NHWC. The final norm of each family is part of
    ``forward_features`` in timm, so the returned map is what timm's own head would pool.
    """

    def __init__(self, model_name: str, *, in_chans: int, img_size: int, pretrained: bool):
        super().__init__()
        import timm

        kwargs: dict[str, Any] = {"num_classes": 0, "in_chans": int(in_chans)}
        if any(token in model_name for token in ("vit", "swin", "deit")):
            kwargs["img_size"] = int(img_size)
        self.load_report: dict[str, Any] | None = None
        if pretrained:
            from source.model.inflate import timm_create_with_report

            self.model, self.load_report = timm_create_with_report(model_name, **kwargs)
        else:
            self.model = timm.create_model(model_name, pretrained=False, **kwargs)
        # Only forward_features is called, so timm's head modules would be parameters without
        # gradients (DDP rejects that). ConvNeXt keeps its final ImageNet LayerNorm in
        # head.norm; it is kept and applied to the map (per position, over channels).
        head_norm = getattr(getattr(self.model, "head", None), "norm", None)
        self.final_norm = head_norm if isinstance(head_norm, nn.Module) and not isinstance(head_norm, nn.Identity) else None
        for attribute in ("head", "fc_norm", "head_drop"):
            if isinstance(getattr(self.model, attribute, None), nn.Module):
                setattr(self.model, attribute, nn.Identity())
        self.model_name = model_name
        self.in_chans = int(in_chans)
        self.num_features = int(self.model.num_features)
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 0) or 0)
        self.pretrained_cfg = dict(getattr(self.model, "pretrained_cfg", {}) or {})

    def forward(self, images: Tensor) -> Tensor:
        features = self.model.forward_features(images)
        if features.ndim == 3:  # ViT tokens [N, prefix + h*w, C]
            tokens = features[:, self.num_prefix_tokens :]
            side = int(round(tokens.shape[1] ** 0.5))
            if side * side != tokens.shape[1]:
                raise RuntimeError(f"{self.model_name}: {tokens.shape[1]} patch tokens are not a square grid")
            return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
        if features.ndim == 4 and features.shape[-1] == self.num_features and features.shape[1] != self.num_features:
            return features.permute(0, 3, 1, 2).contiguous()  # Swin NHWC -> NCHW
        return self.final_norm(features) if self.final_norm is not None else features


def pretrained_summary(backbone: Timm2DBackbone) -> Mapping[str, Any]:
    """Real load result of timm's ImageNet weights, restricted to the tensors kept here."""
    load = dict(backbone.load_report or {})
    own = set(backbone.model.state_dict())
    missing = sorted(set(load.get("missing_keys") or []) & own)
    return {
        "status": "loaded",
        "source": load.get("source", f"timm:{backbone.model_name}"),
        "matched_tensors": len(own) - len(missing),
        "model_tensors": len(own),
        "missing_keys": missing,
        "unexpected_keys": list(load.get("unexpected_keys") or []),
        "shape_mismatch": [],
        "notes": ["timm classifier head dropped"]
        + ([f"RGB stem adapted to {backbone.in_chans} input channel(s) by timm"] if backbone.in_chans != 3 else []),
    }


__all__ = ["Timm2DBackbone", "pretrained_summary"]
