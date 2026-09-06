from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.data.paths import ProjectPaths

COLUMNS = (
    "experiment_id",
    "stage",
    "task",
    "backbone",
    "initialization",
    "source_experiment",
    "dapt",
    "alignment",
    "silver_method",
    "modalities",
    "architecture",
    "peft",
    "gpu_count",
    "strategy",
    "total_params",
    "trainable_params",
    "trainable_percent",
    "gflops",
    "auroc",
    "auroc_ci_low",
    "auroc_ci_high",
    "auprc",
    "auprc_ci_low",
    "auprc_ci_high",
    "brier",
    "dice",
    "nsd",
    "hd95",
    "peak_vram_gb",
    "training_time_min",
    "latency_ms",
    "result_path",
)
FAMILIES = (
    "foundation",
    "dapt",
    "alignment",
    "silver",
    "silver_encoder_adaptation",
    "diagnosis",
    "prognosis",
    "contour",
    "architecture",
    "roi",
    "transfer",
)


def _metric(evaluation: dict[str, Any], name: str, part: str = "value") -> Any:
    value = (evaluation.get("metrics") or {}).get(name)
    if isinstance(value, dict):
        return value.get(part)
    return value if part == "value" else None


def flatten(path: Path, root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment = payload.get("experiment") or {}
    lineage = payload.get("lineage") or {}
    model = payload.get("model") or {}
    compute = payload.get("compute") or {}
    evaluation = payload.get("evaluation") or {}
    return {
        "experiment_id": experiment.get("id"),
        "stage": experiment.get("stage"),
        "task": experiment.get("stage"),
        "backbone": lineage.get("backbone"),
        "initialization": lineage.get("initialization"),
        "source_experiment": lineage.get("source_experiment"),
        "dapt": lineage.get("dapt"),
        "alignment": lineage.get("alignment"),
        "silver_method": lineage.get("silver_method"),
        "modalities": model.get("modalities"),
        "architecture": model.get("architecture"),
        "peft": model.get("peft"),
        "gpu_count": compute.get("gpu_count"),
        "strategy": compute.get("strategy"),
        "total_params": model.get("total_params"),
        "trainable_params": model.get("trainable_params"),
        "trainable_percent": model.get("trainable_percent"),
        "gflops": model.get("gflops_per_volume"),
        "auroc": _metric(evaluation, "auroc"),
        "auroc_ci_low": _metric(evaluation, "auroc", "ci_low"),
        "auroc_ci_high": _metric(evaluation, "auroc", "ci_high"),
        "auprc": _metric(evaluation, "auprc"),
        "auprc_ci_low": _metric(evaluation, "auprc", "ci_low"),
        "auprc_ci_high": _metric(evaluation, "auprc", "ci_high"),
        "brier": _metric(evaluation, "brier"),
        "dice": _metric(evaluation, "dice"),
        "nsd": _metric(evaluation, "nsd"),
        "hd95": _metric(evaluation, "hd95"),
        "peak_vram_gb": compute.get("peak_vram_gb"),
        "training_time_min": compute.get("training_time_min"),
        "latency_ms": compute.get("latency_ms_per_volume"),
        "result_path": str(path.relative_to(root)),
    }


def write(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def family(row: Mapping[str, Any]) -> str:
    stage = str(row.get("stage") or "")
    identifier = str(row.get("experiment_id") or "")
    if stage != "ablation":
        return stage
    if identifier.startswith("A"):
        return "architecture"
    if identifier.startswith(("RM", "RS")):
        return "roi"
    if identifier.startswith("TR"):
        return "transfer"
    return "silver" if identifier.startswith("SA") else "ablation"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build compact paper tables from canonical result.json files"
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = (
        args.output_root.resolve()
        if args.output_root
        else ProjectPaths.resolve().assert_persistent_output()
    )
    results = sorted(path for path in root.rglob("result.json") if "summary" not in path.parts)
    rows = [flatten(path, root) for path in results]
    summary = root / "summary"
    write(rows, summary / "all_runs.csv")
    for name in FAMILIES:
        selected = [row for row in rows if family(row) == name]
        if selected:
            write(selected, summary / f"{name}.csv")
    print(json.dumps({"runs": len(rows), "output": str(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
