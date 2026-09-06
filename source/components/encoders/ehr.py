from __future__ import annotations

from source.clinical.encoder import ClinicalEncoder


class EHREncoder(ClinicalEncoder):
    """Compatibility name for the train-fit clinical MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, dropout, **kwargs)
