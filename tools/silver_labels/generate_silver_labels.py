from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.manifests import read_rows
from source.data.paths import ProjectPaths
from source.data.preflight import require_preflight
from source.distributed.gather import gather_objects
from source.distributed.setup import initialize_distributed, rank_zero_call
from source.engine.experiment import (
    OutputManager,
    atomic_write_bytes,
    atomic_write_json,
    compact_result,
    prepare_resumable_run,
)
from source.silver.falcon import FalconExtractor
from source.silver.generator import SilverGenerator
from source.silver.medgemma import MedGemmaExtractor
from source.silver.qc import silver_qc_summary
from source.silver.schema import TARGETS
from source.utils.config import load_config
from source.utils.console import silver_block
from source.utils.environment import environment_report
from source.utils.logger import RunLogger
from tools._common import import_symbol, select_patient_rows, write_parquet_atomic

STATE_SCHEMA_VERSION = 2


def _provider(
    configuration: dict[str, Any],
    role: str,
    paths: ProjectPaths,
    *,
    local_rank: int | None = None,
):
    role_config = dict(configuration.get(role) or {})
    model_path = paths.code_asset(role_config.get("model_path"))
    model_id = str(model_path or role_config.get("model_id") or "")
    factory_path = str(role_config.get("provider_factory") or "")
    if not model_id or not factory_path:
        raise RuntimeError(f"silver.{role} requires model_path/model_id and provider_factory")
    factory = import_symbol(factory_path)
    provider_kwargs = dict(role_config.get("provider_kwargs") or {})
    if local_rank is not None and provider_kwargs.get("device_map", "auto") == "auto":
        provider_kwargs["device_map"] = {"": local_rank}
    provider = factory(model_id=model_id, **provider_kwargs)
    if getattr(provider, "model_id", None) != model_id:
        raise RuntimeError(f"{role} provider did not preserve exact configured model ID")
    return provider, model_id


def _state_path(run_dir: Path, report_id: str) -> Path:
    token = hashlib.sha256(report_id.encode("utf-8")).hexdigest()[:20]
    return run_dir / "state" / f"{token}.json"


def _cached_report(path: Path, report_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        return None
    rows = payload.get("rows")
    audits = payload.get("audits")
    if not isinstance(rows, list) or len(rows) != len(TARGETS):
        return None
    if not isinstance(audits, list) or len(audits) != len(TARGETS):
        return None
    if any(str(row.get("report_id")) != report_id for row in rows):
        return None
    return [dict(row) for row in rows], [dict(row) for row in audits]


def _generate_report(
    generator: SilverGenerator,
    report: Mapping[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report_id = str(report.get("report_id") or "")
    state_path = _state_path(run_dir, report_id)
    cached = _cached_report(state_path, report_id)
    if cached is not None:
        return cached
    labels, audits = generator.generate_report_with_audit(report)
    rows = [label.as_storage_dict() for label in labels]
    audit_rows = [record.as_dict() for record in audits]
    atomic_write_json(
        state_path,
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "report_id": report_id,
            "rows": rows,
            "audits": audit_rows,
        },
    )
    return rows, audit_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate offline SL00/SL01/SL02 silver labels without expert review"
    )
    parser.add_argument("--config", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--patient-id",
        dest="patient_ids",
        action="append",
        metavar="ID",
        help="process every report for this patient; repeat for more patients",
    )
    selection.add_argument("--max-reports", type=int, help="process the first N reports")
    selection.add_argument(
        "--allow-full",
        action="store_true",
        help="explicit authorization to process every report",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_reports is not None and args.max_reports < 1:
        raise SystemExit("--max-reports must be positive")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    config = load_config(args.config, args.overrides)
    config["overwrite"] = bool(args.overwrite)
    config["resume"] = bool(args.resume or not args.overwrite)
    context = initialize_distributed()
    started = time.perf_counter()
    try:
        paths = ProjectPaths.resolve(config)
        rank_zero_call(context, lambda: require_preflight(config, paths))
        manager = OutputManager(paths)
        experiment_id = str(config["experiment"]["id"])

        def prepare() -> Path:
            destination, _ = prepare_resumable_run(
                manager,
                "silver",
                experiment_id,
                {**config, "resolved_paths": paths.as_dict()},
                overwrite=args.overwrite,
            )
            return destination

        run_dir = rank_zero_call(context, prepare)
        silver_config = dict(config.get("silver") or {})
        method = str(silver_config.get("method") or experiment_id.split("_", 1)[0])
        report_path = Path(str(silver_config.get("reports") or ""))
        if not report_path.is_absolute():
            report_path = paths.dataset_root_for(config) / report_path
        reports = select_patient_rows(read_rows(report_path), args.patient_ids)
        if args.max_reports is not None:
            reports = reports[: args.max_reports]
        if not reports:
            raise RuntimeError("report input is empty")
        report_ids = [str(report.get("report_id") or "") for report in reports]
        if not all(report_ids) or len(report_ids) != len(set(report_ids)):
            raise RuntimeError("report_id values must be non-empty and unique")

        falcon = None
        medgemma = None
        local_rank = context.local_rank if context.distributed and context.device.type == "cuda" else None
        if method.upper() in {"SL01", "SL02"}:
            provider, model_id = _provider(silver_config, "falcon", paths, local_rank=local_rank)
            falcon = FalconExtractor(provider, model_id)
        if method.upper() in {"SL00", "SL02"}:
            provider, model_id = _provider(silver_config, "medgemma", paths, local_rank=local_rank)
            medgemma = MedGemmaExtractor(provider, model_id)
        generator = SilverGenerator(
            method,
            falcon=falcon,
            medgemma=medgemma,
            confidence_threshold=float(silver_config.get("confidence_threshold", 0.8)),
            prompt_version=str(silver_config.get("prompt_version") or "v1"),
            run_id=f"{experiment_id}_{uuid.uuid4().hex[:12]}",
        )
        local_pairs = [
            _generate_report(generator, report, run_dir)
            for report in reports[context.rank :: context.world_size]
        ]
        local_labels = [row for rows, _ in local_pairs for row in rows]
        local_audits = [row for _, audits in local_pairs for row in audits]
        label_shards = gather_objects(local_labels, context)
        audit_shards = gather_objects(local_audits, context)
        if not context.is_main:
            context.barrier()
            return 0
        labels = [row for shard in label_shards or [] for row in shard]
        audits = [row for shard in audit_shards or [] for row in shard]
        labels.sort(key=lambda row: (str(row["report_id"]), str(row["target"])))
        audits.sort(key=lambda row: (str(row["report_id"]), str(row["target"])))
        keys = [(row["report_id"], row["target"]) for row in labels]
        if len(keys) != len(set(keys)) or len(labels) != len(reports) * len(TARGETS):
            raise RuntimeError("silver shard merge has duplicate or missing report targets")
        write_parquet_atomic(labels, run_dir / "labels.parquet")
        audit_bytes = b"\n".join(json.dumps(row, sort_keys=True, default=str).encode("utf-8") for row in audits)
        atomic_write_bytes(run_dir / "audit.jsonl", audit_bytes + b"\n" if audit_bytes else b"")
        report_text_by_id = {str(report.get("report_id")): str(report.get("report_text") or "") for report in reports}
        evaluation = silver_qc_summary(labels, audits, len(reports), reports=report_text_by_id)
        atomic_write_json(run_dir / "qc.json", evaluation)
        result = compact_result(
            config,
            status="completed",
            data={"reports": len(reports)},
            compute={
                "strategy": "report_sharding" if context.world_size > 1 else "single",
                "gpu_count": context.world_size if context.device.type == "cuda" else 0,
                "generation_time_min": (time.perf_counter() - started) / 60,
            },
            evaluation=evaluation,
            reproducibility=environment_report(paths.code_root),
        )
        RunLogger(run_dir / "logs" / "generation.log", echo=False).log(
            f"method={method} reports={len(reports)} accepted={evaluation['accepted']} "
            f"abstained={evaluation['abstained']} no_result={evaluation['no_result']}"
        )
        manager.write_result(run_dir, result)
        print(silver_block(method, evaluation, run_dir))
        context.barrier()
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
