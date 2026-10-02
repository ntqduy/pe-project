"""CT-FM 3D arms of the baseline zoo (LoRA fine-tuning and frozen cached features).

CT-FM's encoder, its strictly-verified weight contract and the feature cache predate this
zoo and are shared with the existing CT-FM runs, so they stay in
``source/components/encoders/image/ct_fm.py`` (registry names ``ct_fm`` and
``ct_fm_features``). This module is the zoo's entry point to them:

    ctfm_lora_3d    backbone ct_fm           image-space SegResEncoder, LoRA on the two
                                             deepest stages (backbones.yaml lora_target_modules)
    ctfm_frozen_3d  backbone ct_fm_features  cached pooled features from
                                             tools/data/build_ctfm_cache.py, encoder frozen

Both then go through the same shared projection + MLP / KAN head as every other arm
(``source/model/classifier.py``).
"""
from __future__ import annotations

from source.components.encoders.image.ct_fm import (  # noqa: F401 - re-exported entry points
    build_ct_fm as build_ctfm_lora_3d,
    build_ct_fm_features as build_ctfm_frozen_3d,
)

__all__ = ["build_ctfm_frozen_3d", "build_ctfm_lora_3d"]
