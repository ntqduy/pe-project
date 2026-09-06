from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn
from torch.nn import functional

from source.components.encoders.image.base import BaseImageEncoder
from source.silver.schema import TARGETS

TARGET_CLASS_COUNTS = {name: 4 if name == "acuity" else 1 for name in TARGETS}
SILVER_STATUSES = ("accepted", "abstained", "no_result")


class SilverEncoderAdaptationModel(nn.Module):
    """C0 image encoder with disposable auxiliary heads for silver adaptation."""

    def __init__(self, image_encoder: BaseImageEncoder, targets: Mapping[str, int]):
        super().__init__()
        if not targets:
            raise ValueError("silver encoder adaptation requires at least one target")
        unknown = set(targets) - set(TARGET_CLASS_COUNTS)
        if unknown:
            raise ValueError(f"unknown silver targets: {sorted(unknown)}")
        invalid = {
            name: classes
            for name, classes in targets.items()
            if int(classes) != TARGET_CLASS_COUNTS[name]
        }
        if invalid:
            raise ValueError(f"invalid silver target class counts: {invalid}")
        self.image_encoder = image_encoder
        self.target_names = tuple(str(name) for name in targets)
        self.target_classes = {str(name): int(classes) for name, classes in targets.items()}
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(image_encoder.feature_dim, classes)
                for name, classes in self.target_classes.items()
            }
        )

    def forward(self, volume: Tensor) -> dict[str, object]:
        features = self.image_encoder.forward_features(volume)
        logits = {name: head(features.global_embedding) for name, head in self.heads.items()}
        return {"logits": logits, "features": features.global_embedding}


def silver_adaptation_loss(
    logits: Mapping[str, Tensor],
    labels: Tensor,
    valid: Tensor,
    target_names: Sequence[str],
    target_weights: Mapping[str, float] | None = None,
) -> Tensor:
    """Target-masked loss normalized per study, then averaged across eligible studies."""

    if labels.ndim != 2 or valid.shape != labels.shape:
        raise ValueError("silver labels and validity must have shape [batch, targets]")
    if labels.shape[1] != len(target_names):
        raise ValueError("silver target order does not match the dataset tensor")
    numerator = labels.new_zeros(labels.shape[0])
    denominator = labels.new_zeros(labels.shape[0])
    for index, name in enumerate(target_names):
        prediction = logits[name]
        selected = valid[:, index].bool()
        weight = float((target_weights or {}).get(name, 1.0))
        if weight < 0:
            raise ValueError(f"silver target weight must be non-negative: {name}={weight}")
        if weight == 0 or not bool(selected.any()):
            continue
        truth = labels[:, index]
        if prediction.shape[-1] == 1:
            loss = functional.binary_cross_entropy_with_logits(
                prediction.squeeze(-1)[selected], truth.float()[selected], reduction="none"
            )
        else:
            loss = functional.cross_entropy(
                prediction[selected], truth.long()[selected], reduction="none"
            )
        numerator[selected] += weight * loss
        denominator[selected] += weight
    eligible = denominator > 0
    if not bool(eligible.any()):
        raise ValueError("batch contains no accepted silver targets with positive weight")
    return (numerator[eligible] / denominator[eligible]).mean()


def summarize_silver_labels(
    silver_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    target_names: Sequence[str],
) -> dict[str, Any]:
    """Summarize actual target-level statuses on fixed train/validation patient splits."""

    selected_targets = tuple(target_names)
    unknown = set(selected_targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"configured targets are absent from the silver schema: {sorted(unknown)}")
    split_by_study: dict[tuple[str, str], str] = {}
    split_studies: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in manifest_rows:
        key = (str(row.get("patient_id") or ""), str(row.get("study_id") or ""))
        split = str(row.get("split") or "")
        if not all(key) or split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid manifest identity/split for silver adaptation: {row}")
        previous = split_by_study.setdefault(key, split)
        if previous != split:
            raise ValueError(f"study appears in multiple splits: {key}")
        split_studies[split].add(key)

    counts = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    per_target: dict[str, Counter[str]] = {name: Counter() for name in selected_targets}
    accepted_studies: dict[str, set[tuple[str, str]]] = defaultdict(set)
    accepted_keys: set[tuple[str, str, str]] = set()
    for row in silver_rows:
        target = str(row.get("target") or "")
        if target not in per_target:
            continue
        key = (str(row.get("patient_id") or ""), str(row.get("study_id") or ""))
        split = split_by_study.get(key)
        if split not in {"train", "validation"}:
            continue
        status = str(row.get("status") or "")
        if status not in SILVER_STATUSES:
            raise ValueError(f"unsupported silver status {status!r} for {key}/{target}")
        counts["eligible"] += 1
        counts[status] += 1
        by_split[split]["eligible"] += 1
        by_split[split][status] += 1
        per_target[target]["eligible"] += 1
        per_target[target][status] += 1
        if status == "accepted":
            accepted_key = (*key, target)
            if accepted_key in accepted_keys:
                raise ValueError(f"multiple accepted silver labels for one study target: {accepted_key}")
            accepted_keys.add(accepted_key)
            accepted_studies[split].add(key)

    eligible = int(counts["eligible"])
    accepted = int(counts["accepted"])
    available_targets = [name for name in selected_targets if per_target[name]["eligible"]]
    return {
        "eligible": eligible,
        "accepted": accepted,
        "abstained": int(counts["abstained"]),
        "no_result": int(counts["no_result"]),
        "effective_coverage": accepted / eligible if eligible else 0.0,
        "available_targets": available_targets,
        "unavailable_targets": [name for name in selected_targets if name not in available_targets],
        "studies": {
            split: {
                "manifest": len(split_studies[split]),
                "with_accepted_target": len(accepted_studies[split]),
            }
            for split in ("train", "validation")
        },
        "by_split": {
            split: {key: int(by_split[split][key]) for key in ("eligible", *SILVER_STATUSES)}
            for split in ("train", "validation")
        },
        "per_target": {
            name: {key: int(per_target[name][key]) for key in ("eligible", *SILVER_STATUSES)}
            for name in selected_targets
        },
    }
