from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from source.components.targets import TargetSpec, normalize_target_spec


ConceptSource = Literal["native", "validated_silver", "expert_reviewed"]
ALLOWED_CONCEPT_SOURCES = {"native", "validated_silver", "expert_reviewed"}


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    source: ConceptSource
    target: str
    target_spec: TargetSpec


def enabled_concept_specs(config: Mapping[str, Any] | None) -> dict[str, ConceptSpec]:
    options = dict(config or {})
    if not bool(options.get("enabled", False)):
        return {}
    raw = options.get("concepts") or {}
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("enabled concept bottleneck requires concept_bottleneck.concepts")
    output: dict[str, ConceptSpec] = {}
    for name, value in raw.items():
        item = dict(value or {})
        if not bool(item.get("enabled", False)):
            continue
        source = str(item.get("source") or "").lower()
        if source not in ALLOWED_CONCEPT_SOURCES:
            raise ValueError(
                f"concept {name!r} needs source native, validated_silver, or expert_reviewed"
            )
        target = str(item.get("target") or "").strip()
        if not target:
            raise ValueError(f"enabled concept {name!r} requires an explicit target")
        target_config = item.get("target_spec") or item.get("type") or "binary"
        if isinstance(target_config, str):
            target_config = {
                "type": target_config,
                **({"classes": item["classes"]} if "classes" in item else {}),
                **({"loss": item["loss"]} if "loss" in item else {}),
            }
        output[str(name)] = ConceptSpec(
            name=str(name),
            source=source,  # type: ignore[arg-type]
            target=target,
            target_spec=normalize_target_spec(target_config),
        )
    if not output:
        raise ValueError("concept bottleneck is enabled but every concept is disabled")
    return output
