"""PENet (Huang et al. 2020) fine-tuned from its released PE weights.

The architecture and weights are loaded exactly as the zero-shot arm does
(``source/components/encoders/image/penet_zeroshot.py:load_penet``: SHA-256 check, strict
load of all 225 tensors). Fine-tuning feeds the whole cached volume through the fully
convolutional encoder instead of 32-slice windows, in PENet's own layout and intensity:

* layout   the RAS cache [x, y, z] becomes PENet's [slices, rows, cols] =
           [z (inferior -> superior, the order the zero-shot slice-order check selected),
           y (anterior -> posterior), x (right -> left)] - the DICOM stacking it was
           trained on. The feature map is mapped back to RAS for Grad-CAM.
* values   HU clipped to [-100, 900], scaled to [0, 1], minus 0.15897 (``penet`` mode)

Its GAP-linear classifier is replaced by the shared projection + head.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from source.model.base import BaselineEncoder, intensity_from_config, resolve_pretrained, scratch_report
from source.model.weights import resolve_weight_path

from ._thirdparty import repo_path

DEFAULT_WEIGHT = "third_party/weights/penet_best.pth.tar"


SLICE_ORDERS = ("inferior_to_superior", "superior_to_inferior")


def ras_to_penet(volume: Tensor, slice_order: str = "inferior_to_superior") -> Tensor:
    """[B, C, x(R), y(A), z(S)] -> [B, C, z, y(A->P), x(R->L)], z in ``slice_order``."""
    values = volume.permute(0, 1, 4, 3, 2).flip(3, 4)
    return values.flip(2) if slice_order == "superior_to_inferior" else values


def penet_to_ras(values: Tensor, slice_order: str = "inferior_to_superior") -> Tensor:
    if slice_order == "superior_to_inferior":
        values = values.flip(2)
    return values.flip(3, 4).permute(0, 1, 4, 3, 2)


class PENetCore(nn.Module):
    def __init__(self, network: nn.Module, slice_order: str = "inferior_to_superior"):
        super().__init__()
        if slice_order not in SLICE_ORDERS:
            raise ValueError(f"PENet slice_order must be one of {SLICE_ORDERS}, got {slice_order!r}")
        self.slice_order = slice_order
        self.net = network
        self.net.classifier = nn.Identity()

    def forward(self, volume: Tensor) -> dict[str, Any]:
        x = ras_to_penet(volume, self.slice_order)
        net = self.net
        if x.size(1) < net.num_channels:
            x = x.expand(-1, net.num_channels // x.size(1), -1, -1, -1)
        x = net.max_pool(net.in_conv(x))
        for encoder in net.encoders:
            x = encoder(x)
        return {"feature_map": penet_to_ras(x, self.slice_order)}


def build_penet_3d(config: Mapping[str, Any]) -> BaselineEncoder:
    import sys

    from source.components.encoders.image.penet_zeroshot import load_penet

    repo = repo_path("penet")
    options = dict(config.get("pretrained") or {})
    checkpoint = resolve_weight_path(options.get("path") or DEFAULT_WEIGHT)
    state: dict[str, Any] = {}

    def load(_options: Mapping[str, Any]) -> dict[str, Any]:
        network, info = load_penet(checkpoint, repo)
        state["network"] = network
        return {
            "status": "loaded",
            "source": f"{checkpoint} (released PENet, SHA-256 verified)",
            "matched_tensors": info["tensors"],
            "model_tensors": info["tensors"],
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatch": [],
            "notes": ["strict load; GAP-linear classifier replaced by the shared head",
                      f"model_args={info['model_args']}"],
        }

    placeholder = BaselineEncoder(nn.Identity(), 2048, "PENet 3D")
    resolve_pretrained(placeholder, config, load)
    network = state.get("network")
    report = placeholder.pretrained_report
    if network is None:
        sys.path.insert(0, str(repo.resolve()))
        try:
            from models.penet_classifier import PENetClassifier  # type: ignore[import-not-found]
        finally:
            sys.path.remove(str(repo.resolve()))
        network = PENetClassifier(model_depth=50, cardinality=32, num_classes=1, num_channels=3, init_method="kaiming")
        report = report if report.get("status") == "scratch" else scratch_report("PENet weights unavailable")
    # The zero-shot run on this cohort (outputs/diagnosis/DX_zeroshot_penet, result.json
    # "slice_order") used inferior_to_superior; keep fine-tuning in the order PENet was scored in.
    core = PENetCore(network, str(config.get("slice_order") or "inferior_to_superior"))
    encoder = BaselineEncoder(
        core,
        int(network.in_channels),
        "PENet 3D",
        intensity=intensity_from_config(config.get("intensity"), default_mode="penet"),
        architecture_note="Model 3D: PENet (ResNeXt-50 3D, SE), cả volume theo layout DICOM của PENet; feature map = encoder cuối",
        # Last encoder's 1x1 convs (conv1, conv3, down_sample); the grouped ResNeXt conv2 is skipped.
        lora_target_modules=tuple(config.get("lora_target_modules") or ("encoders.3",)),
    )
    encoder.pretrained_report = report
    # PENet's bottlenecks keep upstream stochastic depth (a random subset of blocks is skipped
    # per training step), so DDP must tolerate parameters without a gradient in a step.
    encoder.ddp_find_unused_parameters = True
    return encoder
