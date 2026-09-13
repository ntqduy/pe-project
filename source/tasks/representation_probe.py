"""Fixed heads on a frozen, evaluation-mode image encoder."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from source.tasks.diagnosis.heads import DiagnosisHeads
from source.tasks.prognosis.heads import PrognosisHead

# A probe config that does not name its targets stays single-task PE, which is what every
# probe in this repository did before multitask probing existed.
DEFAULT_PROBE_TARGETS: dict[str, int] = {"pe_present": 1}


class RepresentationProbe(nn.Module):
    """Linear evaluation of a frozen representation.

    The encoder is frozen three ways -- no gradients, permanent eval mode, and a
    ``no_grad`` forward -- so the score is a property of the representation rather than of
    the fine-tuning recipe. ``source.engine.factory`` returns this model before
    ``apply_peft`` runs, so ``peft.method`` (including LoRA) never applies to a probe.

    ``targets`` selects single-task or multitask probing for diagnosis: one independent
    linear head per target, identical to the heads the full diagnosis model uses. The
    prognosis probe predicts one endpoint and ignores ``targets``; its head is
    Linear->GELU->Linear, so it is a NON-LINEAR probe and its numbers are comparable only
    with other prognosis probes.
    """

    def __init__(
        self,
        image_encoder,
        task: str,
        targets: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder.requires_grad_(False)
        self.task = task
        self.targets = dict(targets or DEFAULT_PROBE_TARGETS)
        if task == "diagnosis" and not self.targets:
            raise ValueError("a diagnosis probe needs at least one target")
        self.head = (
            DiagnosisHeads(image_encoder.feature_dim, self.targets)
            if task == "diagnosis"
            else PrognosisHead(image_encoder.feature_dim, 64)
        )
        self.image_encoder.eval()

    def train(self, mode=True):
        # Trainer calls .train() every epoch; without this the frozen encoder would still
        # update BatchNorm running statistics and apply Dropout, so "frozen" would not be.
        super().train(mode)
        self.image_encoder.eval()
        return self

    def forward(self, inputs, masks=None):
        volume = inputs["volume"] if self.task == "prognosis" else inputs
        with torch.no_grad():
            spatial = self.image_encoder.forward_features(volume).feature_map
            features = spatial.flatten(2).mean(dim=2)
        return {"logits": self.head(features), "features": features,
                "routing": None, "roi_present": {}}
