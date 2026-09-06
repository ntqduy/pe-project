from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor


def mortality_loss(logits: Tensor, labels: Tensor, valid: Tensor | None = None) -> Tensor:
    selected = valid.bool() if valid is not None else labels.isfinite()
    if not selected.any():
        raise ValueError("batch contains no valid mortality labels")
    return functional.binary_cross_entropy_with_logits(logits[selected], labels.float()[selected])
