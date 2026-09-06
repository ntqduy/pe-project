from __future__ import annotations

from .falcon import FalconExtractor


class MedGemmaExtractor(FalconExtractor):
    """Same strict structured contract, with a separately pinned medical model ID."""
