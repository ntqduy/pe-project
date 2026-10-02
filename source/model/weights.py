"""Where baseline weights live and how a missing file is fetched.

Local weights stay under ``third_party/weights`` (git-ignored, see its README). Files the
baselines download themselves go to ``third_party/weights/baselines/``; timm / MedicalNet
weights use the Hugging Face cache. ``tools/baselines/prepare_weights.py`` fetches all of
them once, so training ranks never race on a download.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from source.data.paths import discover_code_root

# Public URLs of the non-hub checkpoints the baselines use.
SWIN_UNETR_SSL_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
)


def resolve_weight_path(value: str | os.PathLike[str]) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else discover_code_root() / path


def ensure_weight_file(path: str | os.PathLike[str], url: str | None = None, *, allow_download: bool = True) -> Path:
    """Return the local file, downloading ``url`` atomically when it is missing."""
    destination = resolve_weight_path(path)
    if destination.is_file():
        return destination
    if not url or not allow_download:
        raise FileNotFoundError(
            f"pretrained weight not found: {destination}"
            + (f" (download it from {url} or run tools/baselines/prepare_weights.py)" if url else "")
        )
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    print(f"downloading pretrained weight {url} -> {destination}", flush=True)
    torch.hub.download_url_to_file(url, str(temporary), progress=False)
    os.replace(temporary, destination)
    return destination


__all__ = ["SWIN_UNETR_SSL_URL", "ensure_weight_file", "resolve_weight_path"]
