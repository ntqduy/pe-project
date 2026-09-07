"""Patient-level stratified sampling inside the official INSPECT split.

Ported verbatim in behaviour from /mnt/pe_study `create_inspect_subset.py`
(`allocate_counts`, `allocate_total`), which is the logic that produced the
engineering-500 cohort.

Two invariants that make `test_500_sample` a valid rehearsal for `full_inspect`:

1. **Sampling is patient-level.** Patients are drawn, then expanded back to all of
   their studies, so no patient is split across the subset boundary.
2. **The official split is preserved.** Patients are drawn independently inside each
   official split, and the per-split targets are allocated in proportion to the
   official split sizes. A sampled patient never changes split.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .adjudication import PatientLabel
from .sources import OFFICIAL_SPLITS


class SamplingError(ValueError):
    pass


@dataclass(frozen=True)
class SamplingPlan:
    seed: int
    total_patients: int | None
    per_split: dict[str, int]
    strata: tuple[str, ...]
    balance_column: str | None
    positive_fraction: float | None
    selected_patients: int
    strata_summary: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "total_patients": self.total_patients,
            "per_split": self.per_split,
            "strata": list(self.strata),
            "balance_column": self.balance_column,
            "positive_fraction": self.positive_fraction,
            "selected_patients": self.selected_patients,
            "strata_summary": self.strata_summary,
        }


def allocate_total(group_counts: Mapping[str, int], target: int, keys: Sequence[str]) -> dict[str, int]:
    """Allocate ``target`` across groups proportionally, by largest remainder."""
    if target <= 0:
        return {key: 0 for key in keys}
    total = sum(group_counts.get(key, 0) for key in keys)
    if total <= 0:
        raise SamplingError("cannot allocate a sample from empty groups")
    raw = {key: target * group_counts.get(key, 0) / total for key in keys}
    allocation = {key: min(group_counts.get(key, 0), int(raw[key])) for key in keys}
    while sum(allocation.values()) < target:
        candidates = sorted(
            keys, key=lambda key: (raw[key] - allocation[key], group_counts.get(key, 0)), reverse=True
        )
        for key in candidates:
            if allocation[key] < group_counts.get(key, 0):
                allocation[key] += 1
                break
        else:
            break
    return allocation


def allocate_counts(bin_counts: Mapping[tuple[str, ...], int], target: int) -> dict[tuple[str, ...], int]:
    """Allocate ``target`` patients across strata, keeping every non-empty stratum represented."""
    if target <= 0:
        return {}
    total = sum(bin_counts.values())
    if total <= 0:
        raise SamplingError("cannot allocate a sample from empty strata")
    raw = {key: target * count / total for key, count in bin_counts.items()}
    allocation = {key: min(count, int(raw[key])) for key, count in bin_counts.items()}
    if target >= len(bin_counts):
        for key, count in bin_counts.items():
            if count > 0 and allocation[key] == 0:
                allocation[key] = 1
    while sum(allocation.values()) < target:
        candidates = sorted(
            allocation, key=lambda key: (raw[key] - allocation[key], bin_counts[key]), reverse=True
        )
        for key in candidates:
            if allocation[key] < bin_counts[key]:
                allocation[key] += 1
                break
        else:
            break
    while sum(allocation.values()) > target:
        candidates = sorted(
            allocation, key=lambda key: (allocation[key] - raw[key], allocation[key]), reverse=True
        )
        for key in candidates:
            minimum = 1 if target >= len(bin_counts) else 0
            if allocation[key] > minimum:
                allocation[key] -= 1
                break
        else:
            break
    return allocation


def sample_patients(
    labels: Sequence[PatientLabel],
    *,
    seed: int,
    total_patients: int | None = None,
    per_split: Mapping[str, int] | None = None,
    strata: Sequence[str] = (),
    balance_column: str | None = None,
    positive_fraction: float | None = None,
) -> tuple[set[str], SamplingPlan]:
    """Draw a stratified patient sample inside each official split.

    Returns the selected patient ids and the plan that produced them. Give either
    ``total_patients`` (allocated across splits in official proportion) or an explicit
    ``per_split`` mapping.
    """
    if (total_patients is None) == (per_split is None):
        raise SamplingError("give exactly one of sampling.total_patients or sampling.per_split")
    if positive_fraction is not None and not 0 < float(positive_fraction) < 1:
        raise SamplingError("sampling.positive_fraction must be strictly between 0 and 1")
    columns = tuple(strata)
    if positive_fraction is not None:
        if not balance_column:
            raise SamplingError("sampling.positive_fraction requires sampling.balance_column")
        if balance_column not in columns:
            raise SamplingError("sampling.balance_column must also appear in sampling.strata")

    split_counts = Counter(label.split for label in labels)
    if total_patients is not None:
        if int(total_patients) > len(labels):
            raise SamplingError(
                f"requested {total_patients} patients but only {len(labels)} are eligible"
            )
        targets = allocate_total(split_counts, int(total_patients), OFFICIAL_SPLITS)
    else:
        targets = {split: int(count) for split, count in dict(per_split or {}).items()}
    for split, target in targets.items():
        available = split_counts.get(split, 0)
        if target > available:
            raise SamplingError(
                f"requested {target} patients for split={split}, but only {available} are eligible"
            )

    rng = random.Random(int(seed))
    selected: set[str] = set()
    summary: list[dict[str, Any]] = []
    for split in OFFICIAL_SPLITS:
        target = int(targets.get(split, 0))
        split_labels = [label for label in labels if label.split == split]
        by_bin: dict[tuple[str, ...], list[PatientLabel]] = {}
        for label in split_labels:
            by_bin.setdefault(label.stratum(columns), []).append(label)
        bin_counts = Counter({key: len(value) for key, value in by_bin.items()})
        if not bin_counts:
            if target:
                raise SamplingError(f"split={split} has no eligible patients but target={target}")
            continue
        if positive_fraction is None:
            allocation = allocate_counts(bin_counts, target)
        else:
            index = columns.index(str(balance_column))
            class_counts = Counter(key[index] for key, count in bin_counts.items() for _ in range(count))
            true_target = min(round(target * float(positive_fraction)), class_counts["TRUE"])
            false_target = min(target - true_target, class_counts["FALSE"])
            allocation = {}
            for class_value, class_target in (("TRUE", true_target), ("FALSE", false_target)):
                class_bins = Counter(
                    {key: count for key, count in bin_counts.items() if key[index] == class_value}
                )
                if class_bins:
                    allocation.update(allocate_counts(class_bins, class_target))
        for key, count in allocation.items():
            candidates = list(by_bin[key])
            rng.shuffle(candidates)
            selected.update(label.patient_id for label in candidates[:count])
        for key in sorted(by_bin):
            chosen = sum(label.patient_id in selected for label in by_bin[key])
            row = {
                "split": split,
                "stratum": "|".join(key) if key else "(none)",
                "eligible_patients": len(by_bin[key]),
                "sampled_patients": chosen,
            }
            row.update(dict(zip(columns, key)))
            summary.append(row)
        drawn = sum(label.patient_id in selected for label in split_labels)
        if drawn != target:
            raise SamplingError(
                f"sampling produced {drawn} patients for split={split}; expected {target}. "
                "Check the class-balance constraint and the available strata."
            )
    if not selected:
        raise SamplingError("sampling selected no patients; increase the requested counts")
    plan = SamplingPlan(
        seed=int(seed),
        total_patients=int(total_patients) if total_patients is not None else None,
        per_split={split: int(targets.get(split, 0)) for split in OFFICIAL_SPLITS},
        strata=columns,
        balance_column=balance_column,
        positive_fraction=float(positive_fraction) if positive_fraction is not None else None,
        selected_patients=len(selected),
        strata_summary=summary,
    )
    return selected, plan
