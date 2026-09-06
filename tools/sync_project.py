from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.data.paths import ProjectPaths


INCLUDED_FILES = ("README.md", ".gitignore", "pyproject.toml", "requirements.txt")
INCLUDED_DIRECTORIES = ("configs", "source", "tools", "scripts")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "outputs", "cache", "input", ".git"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def candidates(root: Path, include_third_party: bool):
    for name in INCLUDED_FILES:
        path = root / name
        if path.is_file():
            yield path
    directories = list(INCLUDED_DIRECTORIES)
    if include_third_party:
        directories.append("third_party")
    elif (root / "third_party" / "versions.yaml").is_file():
        yield root / "third_party" / "versions.yaml"
    for name in directories:
        directory = root / name
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file() and not (set(path.relative_to(root).parts) & EXCLUDED_PARTS):
                    yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Non-destructively copy project source/config/scripts to cloud pe-project"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform copies; default only prints the plan",
    )
    parser.add_argument("--include-third-party", action="store_true")
    args = parser.parse_args()
    paths = ProjectPaths.resolve()
    if paths.cloud_project_root is None:
        raise SystemExit("set PE_CLOUD_ROOT or PE_CLOUD_PROJECT_ROOT before sync")
    source = paths.code_root
    destination_root = paths.cloud_project_root
    if source.resolve() == destination_root.resolve():
        print("Repository already runs from cloud project root; no sync needed.")
        return 0
    planned = []
    for item in candidates(source, args.include_third_party):
        relative = item.relative_to(source)
        destination = destination_root / relative
        if not destination.is_file() or digest(item) != digest(destination):
            planned.append((item, destination))
    for item, destination in planned:
        print(f"COPY {item.relative_to(source)} -> {destination}")
        if args.execute:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
    print(f"{'Copied' if args.execute else 'Planned'} {len(planned)} files; no cloud files were deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
