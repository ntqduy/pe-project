from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    report: dict[str, Any] = {"seed": seed, "python": True, "numpy": False, "torch": False}
    try:
        import numpy as np

        np.random.seed(seed)
        report["numpy"] = True
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        report["torch"] = True
        report["cuda"] = bool(torch.cuda.is_available())
    except ModuleNotFoundError:
        pass
    return report


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive python/numpy seeds from the worker's torch seed.

    torch already gives each worker ``base_seed + worker_id``; python ``random`` and numpy are
    otherwise forked with the parent's state, so every worker would draw the same numbers.
    """
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_seeding(seed: int) -> dict[str, Any]:
    """``DataLoader(**loader_seeding(seed))``: seeded shuffle order and worker RNG streams."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return {"worker_init_fn": seed_worker, "generator": generator}
