from __future__ import annotations

from typing import Any, Mapping

from .external import InspectedExternalEncoder, build_inspected_external


def build_ct_clip(config: Mapping[str, Any]) -> InspectedExternalEncoder:
    return build_inspected_external(config, "CT-CLIP/CT-RATE")
