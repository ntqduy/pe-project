from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from source.data.paths import discover_code_root

from .base import BaseImageEncoder, ImageFeatures


class ThirdPartyIntegrationError(RuntimeError):
    pass


def import_symbol(path: str) -> Any:
    if ":" not in path:
        raise ThirdPartyIntegrationError(f"symbol must be module:attribute, got {path!r}")
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ThirdPartyIntegrationError(f"missing symbol {path}") from exc


class InspectedExternalEncoder(BaseImageEncoder):
    """Adapter configured only after the concrete third-party API has been inspected.

    The project deliberately does not guess CT-FM, CT-CLIP, or TotalFM call signatures.
    A versioned integration supplies a factory and output adapter in its config.
    """

    def __init__(
        self,
        model: nn.Module,
        output_adapter: Callable[[Any], ImageFeatures | Mapping[str, Tensor]],
        feature_dim: int,
        backbone_name: str,
    ):
        super().__init__()
        self.model = model
        self.output_adapter = output_adapter
        self.feature_dim = int(feature_dim)
        self.backbone_name = backbone_name

    def forward_features(self, volume: Tensor) -> ImageFeatures:
        converted = self.output_adapter(self.model(volume))
        if isinstance(converted, ImageFeatures):
            result = converted
        elif isinstance(converted, Mapping):
            if "feature_map" not in converted or "global_embedding" not in converted:
                raise ThirdPartyIntegrationError("output adapter must return feature_map and global_embedding")
            result = ImageFeatures(
                feature_map=converted["feature_map"],
                global_embedding=converted["global_embedding"],
                pyramid=tuple(converted.get("pyramid", ())),
                metadata=dict(converted.get("metadata", {})),
            )
        else:
            raise ThirdPartyIntegrationError("output adapter returned an unsupported value")
        if result.feature_map.ndim != 5 or result.global_embedding.ndim != 2:
            raise ThirdPartyIntegrationError("adapter contract requires 5D feature map and 2D embedding")
        return result

    def load_pretrained_weights(self, checkpoint: str, strict: bool = True) -> dict[str, Any]:
        source = Path(checkpoint)
        if not source.is_file():
            raise ThirdPartyIntegrationError(f"pretrained checkpoint not found: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping):
            for key in ("model_state", "state_dict", "model"):
                if key in payload and isinstance(payload[key], Mapping):
                    payload = payload[key]
                    break
        if not isinstance(payload, Mapping):
            raise ThirdPartyIntegrationError(f"checkpoint does not contain a state dict: {source}")
        state = {(key.removeprefix("module.")): value for key, value in payload.items()}
        incompatible = self.model.load_state_dict(state, strict=strict)
        return {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }


def build_inspected_external(config: Mapping[str, Any], backbone_name: str) -> InspectedExternalEncoder:
    def local_path(value: Any) -> Path:
        candidate = Path(str(value or ""))
        return candidate if candidate.is_absolute() else discover_code_root() / candidate

    repo = local_path(config.get("repo"))
    checkpoint = local_path(config.get("checkpoint"))
    load_pretrained = bool(config.get("load_pretrained", True))
    factory_path = str(config.get("factory") or "")
    adapter_path = str(config.get("output_adapter") or "")
    feature_dim = int(config.get("feature_dim") or 0)
    missing = [
        name
        for name, value in (
            ("repo", repo.is_dir()),
            ("checkpoint", checkpoint.is_file() or not load_pretrained),
            ("factory", bool(factory_path)),
            ("output_adapter", bool(adapter_path)),
            ("feature_dim", feature_dim > 0),
        )
        if not value
    ]
    if missing:
        raise ThirdPartyIntegrationError(
            f"{backbone_name} integration is incomplete ({', '.join(missing)}); "
            "inspect and pin the actual repository/checkpoint before use"
        )
    sys.path.insert(0, str(repo.resolve()))
    try:
        factory = import_symbol(factory_path)
        output_adapter = import_symbol(adapter_path)
        kwargs = dict(config.get("factory_kwargs") or {})
        model = factory(**kwargs)
    finally:
        if sys.path and sys.path[0] == str(repo.resolve()):
            sys.path.pop(0)
    if not isinstance(model, nn.Module):
        raise ThirdPartyIntegrationError(f"{factory_path} did not return torch.nn.Module")
    wrapper = InspectedExternalEncoder(model, output_adapter, feature_dim, backbone_name)
    if load_pretrained:
        wrapper.load_pretrained_weights(str(checkpoint), strict=bool(config.get("strict_load", True)))
    return wrapper
