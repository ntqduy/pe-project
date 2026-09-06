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
