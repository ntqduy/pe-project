from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from source.roi.counterfactual import MaskingPolicy, apply_mask_transform


def _constant_transform(volume: Tensor, mask: Tensor, fill: float | Tensor, operation: str) -> Tensor:
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    binary = (mask > 0.5).to(volume.device, volume.dtype)
    replacement = torch.as_tensor(fill, device=volume.device, dtype=volume.dtype)
    if operation == "remove":
        return volume * (1 - binary) + replacement * binary
    return volume * binary + replacement * (1 - binary)


def remove_roi(
    volume: Tensor,
    mask: Tensor,
    fill: float | Tensor | None = None,
    *,
    policy: MaskingPolicy | None = None,
) -> Tensor:
    return (
        _constant_transform(volume, mask, fill, "remove")
        if fill is not None
        else apply_mask_transform(volume, mask, operation="remove", policy=policy)
    )


def keep_only_roi(
    volume: Tensor,
    mask: Tensor,
    fill: float | Tensor | None = None,
    *,
    policy: MaskingPolicy | None = None,
) -> Tensor:
    return (
        _constant_transform(volume, mask, fill, "keep")
        if fill is not None
        else apply_mask_transform(volume, mask, operation="keep", policy=policy)
    )


def matched_random_mask(
    mask: Tensor,
    seed: int | None = None,
    *,
    body: Tensor | None = None,
    body_wall: Tensor | None = None,
    patient_ids: Sequence[str] | None = None,
    study_ids: Sequence[str] | None = None,
    control_for: str = "roi",
    return_metadata: bool = False,
) -> Tensor | tuple[Tensor, list[dict[str, Any]]]:
    """Deprecated runtime fallback; use the precomputed, QC-audited ROI8 mask.

    A runtime tensor does not carry physical spacing or the complete forbidden-anatomy
    union, so it cannot satisfy the ROI8 scientific contract. Failing is safer than
    silently generating a different kind of control.
    """

    raise ValueError(
        "matched random controls must be precomputed by data.roi as ROI8; "
        "runtime nearest-voxel controls are prohibited"
    )


def apply_counterfactual(
    volume: Tensor,
    masks: Mapping[str, Tensor],
    specification: str,
    *,
    matched_region: str = "pa",
    fill: float | Tensor | None = None,
    masking_policy: str | Mapping[str, object] = "local_mean",
    seed: int | None = None,
    patient_ids: Sequence[str] | None = None,
    study_ids: Sequence[str] | None = None,
) -> Tensor:
    if not specification or specification == "full":
        return volume
    operation, separator, region = specification.partition("_")
    if isinstance(masking_policy, str):
        policy = MaskingPolicy(name=masking_policy)
    else:
        policy = MaskingPolicy(
            name=str(masking_policy.get("type", masking_policy.get("name", "local_mean"))),
            local_radius_voxels=int(masking_policy.get("local_radius_voxels", 3)),
            noise_scale=float(masking_policy.get("noise_scale", 1.0)),
        )
    if operation == "keep" and specification.startswith("keep_only_"):
        region = specification.removeprefix("keep_only_")
        mask = (
            masks["random"]
            if region == "random" and "random" in masks
            else matched_random_mask(
                masks[matched_region], seed=seed, body=masks.get("body"),
                body_wall=masks.get("body_wall"), patient_ids=patient_ids,
                study_ids=study_ids, control_for=matched_region,
            )
            if region == "random"
            else masks[region]
        )
        return keep_only_roi(volume, mask, fill, policy=policy)
    if operation == "remove" and separator:
        mask = (
            masks["random"]
            if region == "random" and "random" in masks
            else matched_random_mask(
                masks[matched_region], seed=seed, body=masks.get("body"),
                body_wall=masks.get("body_wall"), patient_ids=patient_ids,
                study_ids=study_ids, control_for=matched_region,
            )
            if region == "random"
            else masks[region]
        )
        return remove_roi(volume, mask, fill, policy=policy)
    raise ValueError(f"unsupported ROI counterfactual: {specification}")
