from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ROIDefinition:
    code: str
    name: str
    operation: str

    def filename(self, control_for: str | None = None) -> str:
        if self.code != "ROI8":
            if control_for is not None:
                raise ValueError(f"{self.code} does not accept control_for")
            return f"{self.code}_{self.name}.nii.gz"
        if not control_for:
            raise ValueError("ROI8 requires control_for")
        source = ROI_DEFINITIONS[control_for]
        return f"ROI8_control_for_{source.code}_{source.name}.nii.gz"


# Mirrors the recipes in source.roi.builder. Changing a definition/code is a protocol
# change and requires a new run id; the names make the existing definitions readable.
ROI_DEFINITIONS: dict[str, ROIDefinition] = {
    "ROI1": ROIDefinition("ROI1", "heart_mediastinum", "KEEP_ONLY"),
    "ROI2": ROIDefinition("ROI2", "strict_heart", "KEEP_ONLY"),
    "ROI3": ROIDefinition("ROI3", "dilated_central_pulmonary_artery", "REMOVE_ROI"),
    "ROI4": ROIDefinition("ROI4", "pulmonary_artery_tree", "KEEP_ONLY"),
    "ROI5": ROIDefinition("ROI5", "whole_lung_with_vessels", "KEEP_ONLY"),
    "ROI6": ROIDefinition("ROI6", "lung_parenchyma_without_large_vessels", "KEEP_ONLY"),
    "ROI7": ROIDefinition("ROI7", "heart_hilar_vessels_exclusion", "REMOVE_ROI"),
    "ROI8": ROIDefinition("ROI8", "matched_random_control", "KEEP_ONLY"),
}


def roi_name(code: str, control_for: str | None = None) -> str:
    definition = ROI_DEFINITIONS[code]
    if code != "ROI8":
        return definition.name
    if not control_for:
        raise ValueError("ROI8 requires control_for")
    source = ROI_DEFINITIONS[control_for]
    return f"control_for_{source.code}_{source.name}"


def roi_filename(code: str, control_for: str | None = None) -> str:
    return ROI_DEFINITIONS[code].filename(control_for)

