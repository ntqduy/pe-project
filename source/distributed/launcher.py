from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LaunchSpec:
    command: tuple[str, ...]
    environment: dict[str, str]
    gpu_ids: tuple[int, ...]


def build_launch_spec(entrypoint: str | Path, arguments: Sequence[str], gpu_ids: Sequence[int]) -> LaunchSpec:
    physical = tuple(int(value) for value in gpu_ids)
    if len(physical) != len(set(physical)) or any(value < 0 for value in physical):
        raise ValueError(f"invalid GPU group: {physical}")
    environment = dict(os.environ)
    if physical:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in physical)
    if len(physical) <= 1:
        command = (sys.executable, str(entrypoint), *arguments)
    else:
        torchrun = shutil.which("torchrun")
        if torchrun:
            command = (torchrun, "--standalone", f"--nproc-per-node={len(physical)}", str(entrypoint), *arguments)
        else:
            command = (
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={len(physical)}",
                str(entrypoint),
                *arguments,
            )
    return LaunchSpec(command, environment, physical)


def launch(spec: LaunchSpec, *, check: bool = False, stdout: object = None, stderr: object = None) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        spec.command,
        env=spec.environment,
        text=True,
        stdout=stdout,
        stderr=stderr,
    )
    if check:
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, spec.command)
    return process
