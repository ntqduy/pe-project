from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(2, logits.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    return (1 - (2 * intersection + epsilon) / (denominator + epsilon)).mean()


def contour_loss(logits: Tensor, target: Tensor, bce_weight: float = 0.5) -> Tensor:
    return bce_weight * functional.binary_cross_entropy_with_logits(logits, target.float()) + (1 - bce_weight) * soft_dice_loss(logits, target.float())
