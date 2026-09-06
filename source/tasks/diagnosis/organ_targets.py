from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from source.components.targets import TargetSpec, normalize_target_spec

OrganName = Literal["heart", "pa", "lung"]
SupervisionSource = Literal["native", "silver", "expert_reviewed"]
SUPPORTED_ORGANS = ("heart", "pa", "lung")
SUPPORTED_SOURCES = ("native", "silver", "expert_reviewed")

# Conceptual defaults only. A head is instantiated only when a runnable config explicitly
# names the target and preflight finds its native/expert or accepted-silver supervision.
DEFAULT_ORGAN_TARGET_MAPPING: dict[str, tuple[str, ...]] = {
    "heart": (
        "rv_enlargement",
        "rv_lv_ratio_mentioned",
        "rv_lv_ratio_value",
        "rv_lv_ratio_abnormal",
        "rv_lv_abnormal",
        "septal_bowing",
        "contrast_reflux",
        "reflux",
        "pericardial_effusion",
    ),
    "pa": ("acuity", "central", "lobar", "segmental", "subsegmental", "saddle"),
    "lung": ("pleural_effusion", "chronic_lung_disease", "fibrosis", "emphysema"),
}


@dataclass(frozen=True)
class OrganTarget:
    organ: OrganName
    name: str
    source: SupervisionSource
    spec: TargetSpec


def normalize_organ_target_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    default_source: str = "native",
) -> dict[str, dict[str, OrganTarget]]:
    if not mapping:
        return {}
    if default_source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported auxiliary default source: {default_source}")
    result: dict[str, dict[str, OrganTarget]] = {}
    seen: dict[str, str] = {}
    for organ, targets in mapping.items():
        organ_name = str(organ).lower()
        if organ_name not in SUPPORTED_ORGANS:
            raise ValueError(f"unsupported auxiliary organ: {organ}")
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError(f"auxiliary target mapping for {organ_name} must be non-empty")
        result[organ_name] = {}
        for name, raw in targets.items():
            target_name = str(name)
            if target_name not in DEFAULT_ORGAN_TARGET_MAPPING[organ_name]:
                raise ValueError(
                    f"target {target_name!r} is not validated for the {organ_name} branch; "
                    "update the reviewed organ-target contract before enabling it"
                )
            if target_name in seen:
                raise ValueError(
                    f"auxiliary target {target_name!r} is assigned to both {seen[target_name]} and {organ_name}"
                )
            options = dict(raw) if isinstance(raw, Mapping) else {"classes": raw}
            source = str(options.pop("source", default_source)).lower()
            if source not in SUPPORTED_SOURCES:
                raise ValueError(f"unsupported supervision source for {target_name}: {source}")
            if "type" not in options and "kind" not in options:
                classes = int(options.get("classes", 1))
                options = {**options, "type": "binary" if classes == 1 else "multiclass"}
            spec = normalize_target_spec(options)
            result[organ_name][target_name] = OrganTarget(
                organ=organ_name,  # type: ignore[arg-type]
                name=target_name,
                source=source,  # type: ignore[arg-type]
                spec=spec,
            )
            seen[target_name] = organ_name
    return result


def validate_organ_target_supervision(
    mapping: Mapping[str, Any] | None,
    *,
    native_targets: set[str],
    silver_targets: set[str],
    expert_targets: set[str] | None = None,
    default_source: str = "native",
) -> list[str]:
    errors: list[str] = []
    normalized = normalize_organ_target_mapping(mapping, default_source=default_source)
    available = {
        "native": native_targets,
        "silver": silver_targets,
        "expert_reviewed": set(expert_targets or ()),
    }
    for targets in normalized.values():
        for target in targets.values():
            if target.name not in available[target.source]:
                errors.append(
                    f"{target.organ}.{target.name} requests {target.source} supervision "
                    "but no matching configured target exists"
                )
    return errors
