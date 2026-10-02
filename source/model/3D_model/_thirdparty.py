"""Import helpers for the unmodified upstream repositories under third_party/repos."""
from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from source.data.paths import discover_code_root


def repo_path(name: str) -> Path:
    path = discover_code_root() / "third_party" / "repos" / name
    if not path.is_dir() or not any(path.iterdir()):
        raise ImportError(f"third_party/repos/{name} is missing or empty; run git submodule update --init third_party/repos/{name}")
    return path


@contextlib.contextmanager
def on_path(path: Path) -> Iterator[None]:
    entry = str(path.resolve())
    sys.path.insert(0, entry)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(entry)


def import_from_repo(repo: str, module: str, *, isolate: tuple[str, ...] = ()) -> Any:
    """Import ``module`` from ``third_party/repos/<repo>``.

    ``isolate`` lists top-level module names the repo defines that could collide with
    another repo's (e.g. ``models``, ``util``); cached entries are removed first so the
    right repository's copy is imported.
    """
    for name in isolate:
        for key in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
            del sys.modules[key]
    with on_path(repo_path(repo)):
        return importlib.import_module(module)


def _transformers_generation_aliases() -> None:
    """The Mamba-MAE mamba_ssm fork imports GreedySearch/SampleDecoderOnlyOutput, names newer
    transformers (this project pins >= 4.57) only keep as GenerateDecoderOnlyOutput. Alias
    them before import instead of editing the fork; they are only used for text generation."""
    try:
        import transformers.generation as generation
    except ImportError:
        return
    replacement = getattr(generation, "GenerateDecoderOnlyOutput", None)
    for legacy in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if replacement is not None and not hasattr(generation, legacy):
            setattr(generation, legacy, replacement)


def require_mamba_ssm() -> None:
    _transformers_generation_aliases()
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mamba_ssm (CUDA) is required for the Mamba baselines. Install the Mamba-MAE fork, which "
            "also serves nnMamba and VMamba: pip install -e third_party/repos/mamba_mae/causal-conv1d "
            "&& pip install -e third_party/repos/mamba_mae/mamba2"
        ) from exc
