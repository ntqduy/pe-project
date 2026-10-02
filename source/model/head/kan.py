"""KAN head built from the vendored pykan (third_party/repos/pykan, used unmodified).

pykan's ``MultKAN`` is written for small interactive fits, so three of its defaults are
wrong inside a training loop and are switched off here: ``auto_save`` (writes a checkpoint
directory at construction), ``save_act`` (stores every activation of every step) and
``symbolic_enabled`` (a symbolic branch that is unused but would receive no gradient, which
DDP rejects). Its constructor also reseeds the global RNGs; the caller's RNG state is
restored afterwards so building a KAN head does not change any other initialisation.
"""
from __future__ import annotations

import random
import sys
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from source.data.paths import discover_code_root


def _import_kan():
    try:
        from kan import KAN  # type: ignore[import-not-found]
    except ImportError:
        repo = discover_code_root() / "third_party" / "repos" / "pykan"
        if not (repo / "kan" / "MultKAN.py").is_file():
            raise ImportError(f"pykan not found at {repo}; run git submodule update --init third_party/repos/pykan")
        sys.path.insert(0, str(repo))
        from kan import KAN  # type: ignore[import-not-found]
    return KAN


class KANHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (16,),
        grid: int = 5,
        spline_order: int = 3,
        grid_range: Sequence[float] = (-3.0, 3.0),
        seed: int = 42,
    ):
        super().__init__()
        KAN = _import_kan()
        width = [int(input_dim), *(int(value) for value in hidden_dims), int(output_dim)]
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            self.kan = KAN(
                width=width,
                grid=int(grid),
                k=int(spline_order),
                grid_range=[float(grid_range[0]), float(grid_range[1])],
                symbolic_enabled=False,
                save_act=False,
                auto_save=False,
                seed=int(seed),
                device="cpu",
            )
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        # The symbolic branch is disabled; freeze it so DDP does not wait for its gradients.
        for parameter in self.kan.symbolic_fun.parameters():
            parameter.requires_grad_(False)
        self.width = width

    def forward(self, values: Tensor) -> Tensor:
        # B-spline bases are evaluated in fp32 even under bf16 autocast.
        with torch.autocast(values.device.type, enabled=False):
            output = self.kan(values.float())
        # Do not keep the last batch (and its autograd graph) alive between steps.
        self.kan.cache_data = None
        self.kan.acts = None
        return output
