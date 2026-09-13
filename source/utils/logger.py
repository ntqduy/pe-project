from __future__ import annotations

import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    def __init__(self, path: Path, echo: bool = True):
        self.path = path
        self.echo = echo
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = message.rstrip().splitlines() or [""]
        line = "".join(f"{timestamp} | {item}\n" for item in lines)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            if self.echo:
                print(line, end="", flush=True)

    def exception(self, message: str, error: BaseException) -> None:
        self.log(
            f"{message} | {type(error).__name__}: {error}\n"
            + "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip()
        )
