"""Construction and quality control for the eight support-stage ROI definitions."""

from .builder import build_roi_dataset
from .counterfactual import MaskingPolicy, apply_mask_transform
from .registry import ROI_DEFINITIONS, roi_filename, roi_name

__all__ = [
    "MaskingPolicy",
    "ROI_DEFINITIONS",
    "apply_mask_transform",
    "build_roi_dataset",
    "roi_filename",
    "roi_name",
]
