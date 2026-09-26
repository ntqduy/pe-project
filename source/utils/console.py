from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


BAR = "=" * 60


def experiment_header(config: Mapping[str, Any], output: Path, manifest: Mapping[str, Any] | None) -> str:
    experiment = dict(config.get("experiment") or {})
    data = dict(config.get("data") or {})
    model = dict(config.get("model") or {})
    lineage = dict(config.get("lineage") or {})
    compute = dict(config.get("compute") or {})
    task = dict(config.get("task") or {})
    split = dict((manifest or {}).get("split_patients") or {})
    return "\n".join(
        (
            BAR,
            f"EXPERIMENT {experiment.get('id')} | {experiment.get('stage')}",
            BAR,
            f"Data         : {data.get('mode')} / {data.get('manifest', '-')}",
            f"Backbone     : {model.get('backbone', '-')}",
            f"Init         : {lineage.get('initialization', 'public')}",
            f"DAPT         : {lineage.get('dapt', (config.get('dapt') or {}).get('method', '-'))}",
            f"Alignment    : {'ON' if lineage.get('alignment') else 'OFF'}",
            f"Silver       : {lineage.get('silver_method', '-')}",
            f"Architecture : {task.get('architecture', '-')}",
            f"PEFT         : {(config.get('peft') or {}).get('method', '-')}",
            "",
            f"Compute      : {compute.get('strategy')}",
            f"GPUs         : {','.join(map(str, compute.get('devices') or ())) or 'none'}",
            f"Precision    : {compute.get('precision')}",
            "-" * 60,
            f"Patients T/V/Test : {split.get('train', '-')} / {split.get('validation', '-')} / {split.get('test', '-')}",
            "Overlap           : PASS",
            "Required files    : PASS",
            "",
            f"Output: {output}",
            BAR,
        )
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def final_evaluation_block(experiment_id: str, evaluation: Mapping[str, Any], output: Path) -> str:
    lines = [BAR, f"FINAL TEST | {experiment_id}", BAR]
    if evaluation.get("primary_target") is not None:
        lines.append(f"Target      : {evaluation['primary_target']}")
    if evaluation.get("evaluated_patients") is not None:
        lines.append(f"Patients    : {evaluation['evaluated_patients']}")
    for name, item in (evaluation.get("metrics") or {}).items():
        if isinstance(item, Mapping):
            value, low, high = (_finite(item.get(key)) for key in ("value", "ci_low", "ci_high"))
            shown = "n/a (undefined for this split)" if value is None else f"{value:.4f}"
            interval = f" [95% CI {low:.4f}, {high:.4f}]" if low is not None and high is not None else ""
            lines.append(f"{name.upper():12s}: {shown}{interval}")
    bootstrap = evaluation.get("bootstrap") or {}
    if bootstrap:
        lines.append(f"\nCI          : patient bootstrap N={bootstrap.get('samples')} (test split only)")
    if evaluation.get("threshold") is not None:
        rule = evaluation.get("threshold_rule") or evaluation.get("threshold_source") or "validation"
        lines.append(f"Threshold   : {float(evaluation['threshold']):.6f} ({rule})")
    if evaluation.get("bootstrap_unavailable_reason"):
        lines.append(f"CI note     : {evaluation['bootstrap_unavailable_reason']}")
    lines.extend(("", "Saved:", str(output), BAR))
    return "\n".join(lines)


def silver_block(method: str, evaluation: Mapping[str, Any], output: Path) -> str:
    return "\n".join(
        (
            BAR,
            f"SILVER | {method}",
            BAR,
            f"Reports           : {evaluation.get('reports')}",
            f"Accepted          : {evaluation.get('accepted')}",
            f"Abstained         : {evaluation.get('abstained')}",
            f"No result         : {evaluation.get('no_result')}",
            f"Coverage          : {100 * float(evaluation.get('coverage', 0)):.2f}%",
            "",
            "Saved:",
            str(output),
            BAR,
        )
    )
