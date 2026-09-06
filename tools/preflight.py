from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.data.paths import ProjectPaths
from source.data.preflight import run_preflight

from tools._common import base_parser, resolve_cli_config


def main() -> int:
    parser = base_parser("Validate paths, split, model, supervision, compute, and persistent output")
    args = parser.parse_args()
    try:
        config = resolve_cli_config(args)
        paths = ProjectPaths.resolve(config)
        report = run_preflight(config, paths)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps({"paths": paths.as_dict(), **report.as_dict()}, indent=2))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
