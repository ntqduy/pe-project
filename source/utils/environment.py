from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_state(code_root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> tuple[int, str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(code_root), *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode, result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return 1, ""

    commit_status, commit = command("rev-parse", "HEAD")
    branch_status, branch = command("branch", "--show-current")
    dirty_status, status = command("status", "--porcelain")
    return {
        "git_commit": commit if commit_status == 0 else None,
        "git_dirty": bool(status) if dirty_status == 0 else None,
        "git_branch": branch if branch_status == 0 else None,
    }


def environment_report(code_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        **git_state(code_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
        "gpu_names": [],
        "gpu_count": 0,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        report["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        report["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(report["gpu_count"])]
    except ModuleNotFoundError:
        pass
    return report
