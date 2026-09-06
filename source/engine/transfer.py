from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .checkpoint import load_checkpoint


def transfer_modules(model: Any, checkpoint: str | Path, modules: Iterable[str]) -> dict[str, Any]:
    selected = tuple(modules)
    if not selected:
        raise ValueError("at least one module must be selected for transfer")
    payload = load_checkpoint(checkpoint, model=model, modules=selected, strict=False)
    return {"source": str(Path(checkpoint)), "lineage": payload["lineage"], **payload["load_report"]}
