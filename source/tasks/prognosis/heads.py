from __future__ import annotations

from torch import Tensor, nn


class PrognosisHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, values: Tensor) -> Tensor:
        return super().forward(values).squeeze(-1)


class PrognosisHeads(nn.Module):
    """One binary outcome head per configured prognosis endpoint."""

    def __init__(self, input_dim: int, targets: tuple[str, ...], hidden_dim: int = 64):
        super().__init__()
        if not targets:
            raise ValueError("prognosis requires at least one target")
        self.heads = nn.ModuleDict(
            {name: PrognosisHead(input_dim, hidden_dim=hidden_dim) for name in targets}
        )

    def forward(self, values: Tensor) -> dict[str, Tensor]:
        return {name: head(values) for name, head in self.heads.items()}


__all__ = ["PrognosisHead", "PrognosisHeads"]
