from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from source.roi.counterfactual import MaskingPolicy, apply_mask_transform
from source.roi.random_controls import stable_control_seed


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
    """Exact-volume random control inside a valid body region and outside the target."""

    if body is None:
        raise ValueError("matched random ROI requires a body mask")
    if mask.shape != body.shape:
        raise ValueError("matched ROI and body masks must have identical shapes")
    if body_wall is not None and body_wall.shape != mask.shape:
        raise ValueError("body-wall mask must have the same shape as the matched ROI")
    batch = mask.shape[0]
    if patient_ids is not None and len(patient_ids) != batch:
        raise ValueError("patient_ids length must equal random-mask batch size")
    if study_ids is not None and len(study_ids) != batch:
        raise ValueError("study_ids length must equal random-mask batch size")
    flat = mask.reshape(mask.shape[0], -1)
    flat_body = body.reshape(body.shape[0], -1) > 0.5
    flat_wall = body_wall.reshape(body_wall.shape[0], -1) > 0.5 if body_wall is not None else None
    result = torch.zeros_like(flat)
    metadata: list[dict[str, Any]] = []
    for index, row in enumerate(flat):
        count = int((row > 0.5).sum().item())
        if count <= 0:
            raise ValueError("cannot match an empty target ROI")
        outside_target = row <= 0.5
        body_candidate = flat_body[index] & outside_target
        wall_candidate = flat_wall[index] & body_candidate if flat_wall is not None else None
        if wall_candidate is not None and int(wall_candidate.sum().item()) >= count:
            candidate = wall_candidate
            candidate_name = "body_wall_outside_target"
        elif int(body_candidate.sum().item()) >= count:
            candidate = body_candidate
            candidate_name = "body_outside_target"
        else:
            raise ValueError(
                f"eligible body voxels cannot match target volume: target={count} "
                f"eligible={int(body_candidate.sum().item())}"
            )
        identity_seed = int(seed or 0)
        if patient_ids is not None or study_ids is not None:
            identity_seed = stable_control_seed(
                identity_seed,
                str(patient_ids[index] if patient_ids is not None else index),
                str(study_ids[index] if study_ids is not None else index),
                control_for,
            )
        generator = torch.Generator(device=flat.device).manual_seed(identity_seed)
        eligible = torch.nonzero(candidate, as_tuple=False).flatten()
        coordinates = torch.nonzero(
            candidate.reshape(mask.shape[1:]), as_tuple=False
        ).to(dtype=torch.float32)
        anchor_index = int(
            torch.randint(
                coordinates.shape[0], (1,), generator=generator, device=row.device
            ).item()
        )
        squared_distance = ((coordinates - coordinates[anchor_index]) ** 2).sum(dim=1)
        nearest = torch.argsort(squared_distance)[:count]
        selected = eligible[nearest]
        result[index, selected] = 1
        metadata.append(
            {
                "method": "random_anchor_nearest_eligible_voxels",
                "candidate_region": candidate_name,
                "seed": identity_seed,
                "target_voxels": count,
                "actual_voxels": int(selected.numel()),
                "eligible_voxels": int(eligible.numel()),
                "overlap_voxels": 0,
                "volume_matched": int(selected.numel()) == count,
            }
        )
    output = result.reshape_as(mask)
    return (output, metadata) if return_metadata else output


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
