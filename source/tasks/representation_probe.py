"""Fixed heads on a frozen, evaluation-mode image encoder."""
import torch
from torch import nn

from source.tasks.diagnosis.heads import DiagnosisHeads
from source.tasks.prognosis.heads import PrognosisHead


class RepresentationProbe(nn.Module):
    def __init__(self, image_encoder, task: str):
        super().__init__()
        self.image_encoder = image_encoder.requires_grad_(False)
        self.task = task
        self.head = (DiagnosisHeads(image_encoder.feature_dim, {"pe_present": 1})
                     if task == "diagnosis" else PrognosisHead(image_encoder.feature_dim, 64))
        self.image_encoder.eval()

    def train(self, mode=True):
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
