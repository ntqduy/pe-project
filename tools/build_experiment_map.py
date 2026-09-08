"""Regenerate docs/EXPERIMENT_MAP.md from configs/experiments.yaml.

The registry is the source of truth; the doc is a readable index of it. Run this from
the project root after adding, renaming or re-grouping an experiment:

    python tools/build_experiment_map.py

It reads the file textually rather than through PyYAML so it also works in a bare
checkout with no dependencies installed.
"""
import re
from pathlib import Path

FIELDS = ("config", "group", "legacy_id", "description", "question", "status")
lines = Path("configs/experiments.yaml").read_text().splitlines()
start = next(i for i, line in enumerate(lines) if line.rstrip() == "experiments:")

entries = []
current = None
for raw in lines[start + 1:]:
    name = re.match(r"^  ([a-z][\w.]*):\s*$", raw)
    if name:
        current = {"name": name.group(1)}
        entries.append(current)
        continue
    if current is None:
        continue
    field = re.match(r"^    (\w+):\s*(.*)$", raw)
    if field and field.group(1) in FIELDS:
        value = field.group(2).strip()
        if value in {">-", ">", "|"}:
            value = ""
        current[field.group(1)] = value
    elif current.get("_pending"):
        pass

# Multi-line descriptions folded with >- keep their text on following lines; stitch them.
text = "\n".join(lines[start + 1:])
for entry in entries:
    for field in ("description", "question"):
        if entry.get(field):
            continue
        pattern = re.compile(
            rf"^  {re.escape(entry['name'])}:\s*$(.*?)(?=^  [a-z]|\Z)", re.S | re.M
        )
        block = pattern.search(text)
        if not block:
            continue
        folded = re.search(rf"^    {field}: >-\s*$((?:\n      .*)+)", block.group(1), re.M)
        if folded:
            entry[field] = " ".join(part.strip() for part in folded.group(1).split("\n") if part.strip())

groups: dict[str, list[dict]] = {}
for entry in entries:
    groups.setdefault(entry.get("group", "UNGROUPED"), []).append(entry)

out = [
    "# Experiment map",
    "",
    "One row per registered experiment, generated from",
    "[`configs/experiments.yaml`](../configs/experiments.yaml) — that file is the source of",
    "truth; this table is the readable index. Regenerate it after adding an experiment.",
    "",
    "`name` is what you pass to `run.py` and what a `scripts/` wrapper resolves.",
    "`legacy_id` is the pre-refactor id an arm replaces, kept so older notes stay traceable.",
    "",
    f"{len(entries)} experiments.",
    "",
]
for group, rows in groups.items():
    out += [
        f"## {group}",
        "",
        "| name | config | legacy id | status | what it is |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        out.append(
            "| `{name}` | `{config}` | {legacy} | {status} | {description} |".format(
                name=row["name"],
                config=row.get("config", ""),
                legacy=row.get("legacy_id", "—"),
                status=row.get("status", ""),
                description=row.get("description", "").replace("|", "\\|"),
            )
        )
    out.append("")

Path("docs/EXPERIMENT_MAP.md").write_text("\n".join(out) + "\n")
print(f"wrote docs/EXPERIMENT_MAP.md with {len(entries)} rows in {len(groups)} groups")
