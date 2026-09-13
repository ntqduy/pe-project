from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from source.imaging.nifti import load_nifti, same_geometry, save_binary_mask


class LungMaskIntegrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LungMaskRunner:
    executable: str = "lungmask"
    checkpoint: Path | None = None
    model_name: str = "R231"
    force_cpu: bool = False

    def verify(self) -> str:
        resolved = shutil.which(self.executable)
        if not resolved:
            candidate = Path(self.executable)
            if candidate.is_file():
                resolved = str(candidate.resolve())
        if not resolved:
            raise LungMaskIntegrationError("lungmask executable is unavailable")
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise LungMaskIntegrationError(
                f"explicit LungMask checkpoint is required; automatic download is disabled: {self.checkpoint}"
            )
        return resolved

    def run(
        self,
        input_path: Path,
        destination: Path,
        *,
        gpu_id: int | None,
        log: Callable[[str], None] | None = None,
    ) -> Path:
        executable = self.verify()
        temporary = destination.with_name(f".{destination.name}.lungmask.tmp.nii.gz")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        command = [
            executable,
            str(input_path),
            str(temporary),
            "--modelname",
            self.model_name,
            "--modelpath",
            str(self.checkpoint),
            "--noprogress",
        ]
        if self.force_cpu or gpu_id is None:
            command.append("--cpu")
        environment = dict(os.environ)
        if gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if log:
            log(f"LungMask command={subprocess.list2cmdline(command)}")
        result = subprocess.run(command, check=False, text=True, capture_output=True, env=environment)
        if log and result.stdout.strip():
            log(f"LungMask stdout:\n{result.stdout.strip()}")
        if log and result.stderr.strip():
            log(f"LungMask stderr:\n{result.stderr.strip()}")
        if result.returncode:
            temporary.unlink(missing_ok=True)
            detail = result.stderr.strip() or result.stdout.strip() or f"exit_code={result.returncode}"
            raise LungMaskIntegrationError(detail[-2000:])
        array, mask_image = load_nifti(temporary)
        _, reference = load_nifti(input_path)
        if not same_geometry(mask_image, reference):
            temporary.unlink(missing_ok=True)
            raise LungMaskIntegrationError("LungMask output geometry does not match the input volume")
        save_binary_mask(array > 0, reference, destination)
        temporary.unlink(missing_ok=True)
        return destination
