"""Construction and quality control for the eight support-stage ROI definitions."""

from .builder import build_roi_dataset, compact_roi_manifest
from .counterfactual import MaskingPolicy, apply_mask_transform

__all__ = ["MaskingPolicy", "apply_mask_transform", "build_roi_dataset", "compact_roi_manifest"]
