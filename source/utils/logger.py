from __future__ import annotations

import os
from pathlib import Path


class RunLogger:
    def __init__(self, path: Path, echo: bool = True):
        self.path = path
        self.echo = echo
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = message.rstrip() + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        if self.echo:
            print(line, end="", flush=True)
