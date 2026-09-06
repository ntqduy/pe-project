from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source.data.manifests import create_patient_split, read_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly create one deterministic patient-level train/validation/test split"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--create-split", action="store_true", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(
            f"split already exists: {args.output}; use --force explicitly to replace"
        )
    rows = create_patient_split(read_rows(args.input), args.seed, tuple(args.ratios))
    if not rows:
        raise SystemExit("input manifest is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created patient-level split with {len(rows)} studies at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
