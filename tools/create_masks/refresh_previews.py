"""Redraw existing segmentation contact sheets from cached masks, without segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from source.data.manifests import read_rows
from source.imaging.preview import write_segmentation_contact_sheet
from source.segmentation.labels import study_pe_labels
from source.segmentation.pipeline import study_preview_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--diagnosis-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="redraw only the first N studies")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    segmentation = config.get("segmentation") or {}
    labels = study_pe_labels(read_rows(args.diagnosis_manifest))
    state_files = sorted((run_dir / "state").glob("*/*.json"))
    if not state_files:
        parser.error(f"no cached segmentation studies under {run_dir / 'state'}")
    if args.limit is not None:
        state_files = state_files[:args.limit]

    counts = {"positive": 0, "negative": 0, "unknown": 0}
    for state_file in state_files:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        patient_id = str(state["patient_id"])
        study_id = str(state["study_id"])
        rows = state["rows"]
        image_path = Path(str(rows[0]["image_path"]))
        label = labels.get((patient_id, study_id))
        counts["unknown" if label is None else "positive" if label else "negative"] += 1
        destination = study_preview_path(run_dir, patient_id, study_id)
        write_segmentation_contact_sheet(
            image_path,
            rows,
            destination,
            study_id=study_id,
            patient_id=patient_id,
            window_width=float(segmentation.get("preview_window_width", 700.0)),
            window_level=float(segmentation.get("preview_window_level", 100.0)),
            pe_present=label,
        )
        print(f"refreshed {destination} | PE: {'Có' if label else 'Không' if label is False else 'Không rõ'}", flush=True)
    print(f"done: {len(state_files)} previews, {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
