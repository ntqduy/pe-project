from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from source.components.targets import TargetSpec

from .schema import ConceptSpec


class ConceptBottleneck(nn.Module):
    """Predicts only explicitly defensible concepts and embeds their predictions."""

    def __init__(
        self,
        input_dim: int,
        concepts: Mapping[str, ConceptSpec],
        output_dim: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not concepts:
            raise ValueError("concept bottleneck requires at least one enabled concept")
        self.concepts = dict(concepts)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, concept.target_spec.output_dim),
                )
                for name, concept in self.concepts.items()
            }
        )
        encoded_dim = sum(concept.target_spec.output_dim for concept in self.concepts.values())
        self.projection = nn.Sequential(
            nn.Linear(encoded_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = int(output_dim)

    @property
    def target_specs(self) -> dict[str, TargetSpec]:
        return {name: concept.target_spec for name, concept in self.concepts.items()}

    @property
    def target_columns(self) -> dict[str, str]:
        return {name: concept.target for name, concept in self.concepts.items()}

    def forward(self, features: Tensor) -> dict[str, object]:
        logits = {name: head(features) for name, head in self.heads.items()}
        values = []
        for name, concept in self.concepts.items():
            prediction = logits[name]
            if concept.target_spec.kind == "binary":
                prediction = torch.sigmoid(prediction)
            elif concept.target_spec.kind == "multiclass":
                prediction = torch.softmax(prediction, dim=-1)
            values.append(prediction)
        return {"logits": logits, "embedding": self.projection(torch.cat(values, dim=1))}
