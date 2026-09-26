from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .base import ImageFeatures
from .external import InspectedExternalEncoder, ThirdPartyIntegrationError, build_inspected_external


def build_ct_fm(config: Mapping[str, Any]) -> InspectedExternalEncoder:
    return build_inspected_external(config, "CT-FM")


# CT-FM was pre-trained on SPL-oriented volumes with HU clipped to [-1024, 2048] and scaled
# to [0, 1] (lighter_zoo preprocessing). The shared image cache stores RAS arrays min-max
# scaled from its own HU window, so an image-space CT-FM run converts both on the fly.
CTFM_ORIENTATION = "SPL"
CTFM_HU_RANGE = (-1024.0, 2048.0)
# axis -> (permutation source axis, flip) taking an RAS [x, y, z] array to SPL [z, y, x]:
# S = +z, P = -y, L = -x.
_RAS_TO_SPL_PERMUTE = (0, 1, 4, 3, 2)
_RAS_TO_SPL_FLIP = (3, 4)


def ras_to_spl(volume: Tensor) -> Tensor:
    """[B, C, x(R), y(A), z(S)] -> [B, C, S, P, L]."""
    return volume.permute(*_RAS_TO_SPL_PERMUTE).flip(_RAS_TO_SPL_FLIP)


def spl_to_ras(volume: Tensor) -> Tensor:
    """Inverse of :func:`ras_to_spl`, applied to every level of the feature pyramid."""
    return volume.flip(_RAS_TO_SPL_FLIP).permute(*_RAS_TO_SPL_PERMUTE)


def remap_intensity(volume: Tensor, input_hu_range: Sequence[float]) -> Tensor:
    """Min-max values of `input_hu_range` -> CT-FM's (HU + 1024) / 3072 scaling.

    HU above the input window's upper bound was already clipped by the cache and cannot be
    recovered; the mapping is exact inside the window.
    """
    low, high = (float(value) for value in input_hu_range)
    target_low, target_high = CTFM_HU_RANGE
    hounsfield = volume * (high - low) + low
    return ((hounsfield - target_low) / (target_high - target_low)).clamp(0.0, 1.0)


def _input_contract_encoder(base: type[nn.Module]) -> type[nn.Module]:
    """SegResEncoder subclass that converts its input to the CT-FM contract.

    A subclass instead of a wrapper keeps the state-dict keys identical to the upstream
    checkpoint (strict load of all 161 tensors) and adds no parameters or buffers. Features
    are returned in the input's axis order, so region masks and ROI counterfactuals, which
    live on the cached RAS grid, still line up with the feature map.
    """

    class CTFMInputContractEncoder(base):  # type: ignore[misc, valid-type]
        def __init__(self, *, input_orientation: str, input_hu_range: Sequence[float] | None, **kwargs: Any):
            super().__init__(**kwargs)
            orientation = str(input_orientation).upper()
            if orientation not in {"RAS", CTFM_ORIENTATION}:
                raise ThirdPartyIntegrationError(f"CT-FM input orientation {orientation!r} is not supported")
            self.input_orientation = orientation
            self.input_hu_range = tuple(float(value) for value in input_hu_range) if input_hu_range else None

        def forward(self, volume: Tensor) -> list[Tensor]:
            if self.input_hu_range is not None:
                volume = remap_intensity(volume, self.input_hu_range)
            if self.input_orientation == CTFM_ORIENTATION:
                return super().forward(volume)
            return [spl_to_ras(level) for level in super().forward(ras_to_spl(volume))]

    return CTFMInputContractEncoder


def build_ct_fm_backbone(
    spatial_dims: int = 3,
    init_filters: int = 32,
    in_channels: int = 1,
    blocks_down: Sequence[int] = (1, 2, 2, 4, 4),
    input_orientation: str | None = None,
    input_hu_range: Sequence[float] | None = None,
) -> nn.Module:
    """SegResEncoder matching project-lighter/ct_fm_feature_extractor.

    Defaults are the values read from that checkpoint's own config.json, verified by a strict
    load of all 161 tensors. `head_module` is deliberately left unset: upstream's
    scripts/feature_extractor.py pools inside the model, which would discard the feature map
    the project contract requires. The adapter below pools instead.

    `input_orientation` / `input_hu_range` describe the arrays the dataset feeds in (the
    shared cache: RAS, min-max of [-1000, 1000] HU). When given, the encoder reorients them to
    SPL and rescales them to CT-FM's HU scaling before the first convolution. Spacing is not
    converted: the shared cache is ~2.4 x 2.4 x 2.0 mm, not CT-FM's 3 x 1 x 1 mm.
    """
    from monai.networks.nets.segresnet_ds import SegResEncoder

    kwargs = {
        "spatial_dims": int(spatial_dims),
        "init_filters": int(init_filters),
        "in_channels": int(in_channels),
        "blocks_down": tuple(int(value) for value in blocks_down),
    }
    if input_orientation is None and input_hu_range is None:
        return SegResEncoder(**kwargs)
    return _input_contract_encoder(SegResEncoder)(
        input_orientation=input_orientation or CTFM_ORIENTATION,
        input_hu_range=input_hu_range,
        **kwargs,
    )


def ct_fm_output_adapter(outputs: Any) -> ImageFeatures:
    if isinstance(outputs, Tensor):
        pyramid: tuple[Tensor, ...] = (outputs,)
    elif isinstance(outputs, (list, tuple)) and outputs:
        pyramid = tuple(outputs)
    else:
        raise ThirdPartyIntegrationError("CT-FM encoder must return a tensor or a non-empty feature pyramid")
    feature_map = pyramid[-1]
    global_embedding = torch.nn.functional.adaptive_avg_pool3d(feature_map, 1).flatten(start_dim=1)
    return ImageFeatures(
        feature_map=feature_map,
        global_embedding=global_embedding,
        pyramid=pyramid,
        metadata={"backbone": "ct_fm", "pooling": "adaptive_avg_pool3d"},
    )


# ---------------------------------------------------------------------------------------
# Precomputed ("cached") CT-FM features
#
# CT-FM was pretrained on SPL volumes resampled to 3 x 1 x 1 mm and consumed in
# 24 x 128 x 128 patches (third_party/repos/CT-FM/scripts/feature_extractor.py). Squeezing a
# whole chest into one 24 x 128 x 128 tensor gives ~12 x 1.9 x 2.6 mm voxels instead, a
# scale the model never saw. The frozen-CT-FM runs therefore extract features once per study
# at the upstream contract (tools/data/build_ctfm_cache.py) and train on those features.
# ---------------------------------------------------------------------------------------

CTFM_SPACING_MM = (3.0, 1.0, 1.0)
CTFM_PATCH_SIZE = (24, 128, 128)
CTFM_FEATURE_REPRESENTATION = "ct_fm_features_v1"


@torch.no_grad()
def ctfm_patch_features(
    model: nn.Module,
    volume: Tensor,
    *,
    patch_size: Sequence[int] = CTFM_PATCH_SIZE,
    batch_size: int = 16,
) -> Tensor:
    """Deepest-level CT-FM feature map of a [1, 1, D, H, W] volume, patch by patch.

    The volume is tiled into non-overlapping ``patch_size`` patches exactly like upstream's
    ``SlidingWindowSplitter(patch_size, 0.0)``; each patch's deepest feature map is written
    back at its tile position, giving one [1, 512, d, h, w] map for the whole volume.
    D, H and W must be multiples of the patch size (the cache canvas guarantees this).
    """
    if volume.ndim != 5 or volume.shape[:2] != (1, 1):
        raise ValueError(f"expected a [1, 1, D, H, W] volume, got {tuple(volume.shape)}")
    patch = tuple(int(value) for value in patch_size)
    spatial = tuple(int(value) for value in volume.shape[-3:])
    if any(size % step for size, step in zip(spatial, patch)):
        raise ValueError(f"volume {spatial} is not a multiple of the CT-FM patch {patch}")
    grid = tuple(size // step for size, step in zip(spatial, patch))
    origins = [
        (i * patch[0], j * patch[1], k * patch[2])
        for i in range(grid[0])
        for j in range(grid[1])
        for k in range(grid[2])
    ]
    output: Tensor | None = None
    cell: tuple[int, int, int] | None = None
    for start in range(0, len(origins), max(1, int(batch_size))):
        chunk = origins[start : start + max(1, int(batch_size))]
        batch = torch.cat(
            [
                volume[..., z : z + patch[0], y : y + patch[1], x : x + patch[2]]
                for z, y, x in chunk
            ],
            dim=0,
        )
        features = model(batch)
        deepest = features[-1] if isinstance(features, (list, tuple)) else features
        if output is None:
            cell = tuple(int(value) for value in deepest.shape[-3:])
            output = volume.new_zeros(
                (1, int(deepest.shape[1]), grid[0] * cell[0], grid[1] * cell[1], grid[2] * cell[2]),
                dtype=torch.float32,
            )
        assert cell is not None
        for (z, y, x), value in zip(chunk, deepest):
            i, j, k = z // patch[0], y // patch[1], x // patch[2]
            output[
                0, :,
                i * cell[0] : (i + 1) * cell[0],
                j * cell[1] : (j + 1) * cell[1],
                k * cell[2] : (k + 1) * cell[2],
            ] = value.float()
    assert output is not None
    return output


class CTFMFeaturePassthrough(nn.Module):
    """The "backbone" of a cached-feature run: splits the cached tensor, computes nothing.

    Cached tensors are [B, 513, d, h, w]: 512 CT-FM feature channels plus one channel with
    the fraction of each feature cell that lies inside the scanned body box (0 in padding).
    """

    def forward(self, cached: Tensor) -> dict[str, Tensor]:
        if cached.ndim != 5 or cached.shape[1] != 513:
            raise ThirdPartyIntegrationError(
                "cached CT-FM input must be [B, 512 + 1, d, h, w]; got "
                f"{tuple(cached.shape)}. Point data.manifest at manifests/ct_fm/*.csv."
            )
        return {"feature_map": cached[:, :-1], "valid": cached[:, -1:]}


def build_ct_fm_feature_passthrough(**_: Any) -> nn.Module:
    return CTFMFeaturePassthrough()


def ct_fm_cached_feature_adapter(outputs: Any) -> ImageFeatures:
    """Global embedding = mean over feature cells inside the body box (padding excluded)."""
    if not isinstance(outputs, Mapping) or "feature_map" not in outputs:
        raise ThirdPartyIntegrationError("cached CT-FM passthrough must return feature_map and valid")
    feature_map = outputs["feature_map"]
    weights = outputs["valid"].clamp(0.0, 1.0).to(feature_map.dtype)
    denominator = weights.sum(dim=(2, 3, 4)).clamp_min(1e-6)
    global_embedding = (feature_map * weights).sum(dim=(2, 3, 4)) / denominator
    return ImageFeatures(
        feature_map=feature_map,
        global_embedding=global_embedding,
        pyramid=(feature_map,),
        metadata={"backbone": "ct_fm_features", "pooling": "body_weighted_mean"},
    )


class CachedCTFMEncoder(InspectedExternalEncoder):
    """Encoder over precomputed CT-FM features; the weights were applied when caching."""

    def load_pretrained_weights(self, checkpoint: str, strict: bool = True) -> dict[str, Any]:
        from pathlib import Path

        if not Path(checkpoint).is_file():
            raise ThirdPartyIntegrationError(f"CT-FM checkpoint not found: {checkpoint}")
        return {
            "missing_keys": [],
            "unexpected_keys": [],
            "note": "features were extracted with this checkpoint by tools/data/build_ctfm_cache.py",
        }


def build_ct_fm_features(config: Mapping[str, Any]) -> CachedCTFMEncoder:
    from pathlib import Path

    from source.data.paths import discover_code_root

    feature_dim = int(config.get("feature_dim") or 512)
    encoder = CachedCTFMEncoder(
        CTFMFeaturePassthrough(), ct_fm_cached_feature_adapter, feature_dim, "CT-FM (cached features)"
    )
    if bool(config.get("load_pretrained", True)):
        checkpoint = Path(str(config.get("checkpoint") or ""))
        if not checkpoint.is_absolute():
            checkpoint = discover_code_root() / checkpoint
        encoder.load_pretrained_weights(str(checkpoint))
    return encoder
