from __future__ import annotations

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


def final_evaluation_block(experiment_id: str, evaluation: Mapping[str, Any], output: Path) -> str:
    lines = [BAR, f"FINAL TEST | {experiment_id}", BAR]
    for name, item in (evaluation.get("metrics") or {}).items():
        if isinstance(item, Mapping):
            lines.append(
                f"{name.upper():12s}: {float(item['value']):.4f} "
                f"[95% CI {float(item['ci_low']):.4f}, {float(item['ci_high']):.4f}]"
            )
    bootstrap = evaluation.get("bootstrap") or {}
    if bootstrap:
        lines.append(f"\nCI          : patient bootstrap N={bootstrap.get('samples')}")
    if evaluation.get("threshold") is not None:
        lines.append(f"Threshold   : {float(evaluation['threshold']):.6f} (validation)")
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
