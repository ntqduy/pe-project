"""CLI: exploratory data analysis for one dataset profile.

    python analysis/run_eda.py --profile test_500_sample
    python analysis/run_eda.py --profile full_inspect --rule-sample 5000

Writes one run directory under <output_root>/EDA/<profile>/ and never touches anything a
training stage reads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import cohort, figures, geometry, labels, outcomes, report, reports
from analysis.loaders import load_dataset
from source.data.paths import PathConfigurationError, ProjectPaths
from source.utils.logger import RunLogger

DEFAULT_PROFILE = "test_500_sample"


def _csv(rows: list[dict[str, Any]], destination: Path) -> None:
    """Flat CSV without pandas: the EDA must run on a bare interpreter too."""
    import csv

    if not rows:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _label_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "available":
        return []
    rows = [{"scope": "overall", "label": name, **values}
            for name, values in (payload.get("overall") or {}).items()]
    for split, per_label in (payload.get("by_split") or {}).items():
        rows += [{"scope": f"split:{split}", "label": name, **values}
                 for name, values in per_label.items()]
    return rows


def _outcome_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest, values in payload.items():
        if not isinstance(values, dict) or values.get("status") != "available":
            continue
        for endpoint, summary in values["endpoints"].items():
            rows.append({
                "manifest": manifest,
                "endpoint": endpoint,
                "events": summary["events"],
                "non_events": summary["non_events"],
                "missing": summary["missing"],
                "observed": summary["observed"],
                "event_rate_of_observed": summary["event_rate_of_observed"],
                "minimum_split_events": summary["minimum_split_events"],
                "recommended_protocol": summary["recommended_protocol"],
            })
    return rows


def _cohort_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest, values in (payload.get("per_manifest") or {}).items():
        rows.append({
            "manifest": manifest, "scope": "total",
            "patients": values["patients"], "studies": values["studies"],
        })
        for split, counts in (values.get("splits") or {}).items():
            rows.append({
                "manifest": manifest, "scope": f"split:{split}",
                "patients": counts["patients"], "studies": counts["studies"],
            })
    return rows


def _rule_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = (payload or {}).get("rule_coverage") or {}
    if coverage.get("status") != "available":
        return []
    return [
        {"target": target, "resolved": values["resolved"],
         "resolved_rate": values["resolved_rate"],
         "reports_examined": coverage["reports_examined"]}
        for target, values in coverage["by_target"].items()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analysis/run_eda.py",
        description="Describe one built dataset profile: cohort, labels, endpoints, geometry, reports.",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE,
                        help=f"dataset profile to describe (default: {DEFAULT_PROFILE})")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="override the resolved dataset root")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="override the resolved output root; EDA/<profile>/ is created under it")
    parser.add_argument("--rule-sample", type=int, default=2000,
                        help="reports scanned for rule coverage; 0 = every report")
    parser.add_argument("--no-figures", action="store_true", help="skip PNG rendering")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()

    try:
        paths = ProjectPaths.resolve({})
    except PathConfigurationError as exc:
        print(f"error: cannot resolve project paths ({exc}); set PE_CLOUD_ROOT", file=sys.stderr)
        return 2

    dataset_root = args.dataset_root
    if dataset_root is None:
        if paths.derived_root is None:
            print("error: derived data root is not configured; set PE_CLOUD_ROOT or "
                  "PE_DERIVED_ROOT, or pass --dataset-root", file=sys.stderr)
            return 2
        dataset_root = paths.derived_root / "datasets" / args.profile
    dataset_root = dataset_root.resolve()

    output_root = args.output_root or paths.output_root
    if output_root is None:
        print("error: output root is not configured; set PE_CLOUD_ROOT or pass --output-root",
              file=sys.stderr)
        return 2
    output_dir = (Path(output_root) / "EDA" / args.profile).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(output_dir / "logs" / "run.log")
    logger.log(f"eda profile={args.profile} status=started")
    logger.log(f"command={' '.join(sys.argv)}")
    logger.log(f"dataset_root={dataset_root}")
    logger.log(f"output_dir={output_dir}")

    if not dataset_root.is_dir():
        logger.log(f"eda status=failed reason=dataset_root_not_found path={dataset_root}")
        print(f"error: dataset root does not exist: {dataset_root}\n"
              f"       build it first, e.g. bash scripts/0_data_preprocessing/build_{args.profile}.sh",
              file=sys.stderr)
        return 3

    try:
        bundle = load_dataset(dataset_root, args.profile)
        logger.log(f"tables_present={sorted(bundle.tables)}")
        if bundle.missing:
            logger.log(f"tables_missing={sorted(bundle.missing)}")
        if bundle.unreadable:
            logger.log(f"tables_unreadable={bundle.unreadable}")

        sample = None if int(args.rule_sample) <= 0 else int(args.rule_sample)
        summary: dict[str, Any] = {
            "profile": args.profile,
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inventory": bundle.as_inventory(),
            "provenance": bundle.provenance,
            "cohort": cohort.analyse(bundle),
            "labels": labels.analyse(bundle),
            "outcomes": outcomes.analyse(bundle),
            "geometry": geometry.analyse(bundle),
            "reports": reports.analyse(bundle, sample),
        }
        for section in ("cohort", "labels", "outcomes", "geometry", "reports"):
            logger.log(f"section={section} done")

        summary["figures"] = (
            {"status": "skipped", "reason": "--no-figures"}
            if args.no_figures
            else figures.render_all(summary, output_dir / "figures")
        )

        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        _csv(bundle.as_inventory(), output_dir / "inventory.csv")
        _csv(_cohort_rows(summary["cohort"]), output_dir / "cohort.csv")
        _csv(_label_rows(summary["labels"]), output_dir / "labels.csv")
        _csv(_outcome_rows(summary["outcomes"]), output_dir / "outcomes.csv")
        _csv(_rule_rows(summary["reports"]), output_dir / "rule_coverage.csv")
        report.write(summary, output_dir / "EDA_REPORT.md")

        logger.log(f"eda status=finished elapsed_sec={time.perf_counter() - started:.2f}")
        print(f"EDA written to {output_dir}")
        print(f"  read first: {output_dir / 'EDA_REPORT.md'}")
        return 0
    except Exception as exc:  # noqa: BLE001 - report the traceback into the run log
        logger.log(f"eda status=failed reason={type(exc).__name__}: {exc}")
        logger.log(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
