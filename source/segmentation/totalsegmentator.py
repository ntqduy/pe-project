from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from source.imaging.morphology import boundary_shell, physical_dilation
from source.imaging.nifti import combine_masks, load_image, load_nifti, same_geometry, save_binary_mask


class SegmentationIntegrationError(RuntimeError):
    pass


TASK_CLASSES = {
    "total": (
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
        "heart",
    ),
    "trunk_cavities": ("mediastinum",),
    "heartchambers_highres": (
        "heart_myocardium",
        "heart_ventricle_right",
        "heart_ventricle_left",
        "heart_atrium_right",
        "heart_atrium_left",
        "pulmonary_artery",
    ),
    "lung_vessels": ("lung_airways", "lung_arteries", "lung_veins"),
    "body": ("body_trunc", "body_extremities"),
}
# The five lobes come out of the same ``total`` run that builds ``lung``. They are kept, with
# both lungs, because radiology reports localize a PE by side and lobe (INSPECT impressions:
# side in ~88%, lobe in ~62% of PE-positive reports).
LUNG_LOBES = tuple(name for name in TASK_CLASSES["total"] if name.startswith("lung_"))
LUNG_SIDES = {
    "lung_left": tuple(name for name in LUNG_LOBES if name.endswith("_left")),
    "lung_right": tuple(name for name in LUNG_LOBES if name.endswith("_right")),
}
# Directories written by the previous output layout inside masks/<patient>/<study>/.
# ``tasks`` held raw TotalSegmentator output, ``canonical`` the project masks.
LEGACY_STUDY_DIRECTORIES = ("tasks", "canonical")
# Dataset IDs from the pinned TotalSegmentator map_tasks_config.py. Check the actual
# nnU-Net checkpoint tree before running, so an empty directory cannot trigger downloads.
TASK_WEIGHT_IDS = {
    "trunk_cavities": (343,),
    "heartchambers_highres": (301,),
    "lung_vessels": (117,),
    "body": (299,),
}


@dataclass(frozen=True)
class TotalSegmentatorRunner:
    executable: str = "TotalSegmentator"
    repository: Path | None = None
    weights_directory: Path | None = None
    fast: bool = False
    body_wall_thickness_mm: float = 15.0
    hilar_proximity_mm: float = 12.0
    extra_arguments: tuple[str, ...] = ()
    # Raw per-task TotalSegmentator output is an intermediate. It is written to a local
    # temporary directory (system temp when None) and deleted after the masks are built,
    # so it never reaches the persistent output tree.
    scratch_directory: Path | None = None
    # TotalSegmentator defaults to 6 saving processes of several GB each; on a 16 GB VM the
    # kernel OOM-kills them and nnU-Net then waits forever for the dead workers.
    resample_threads: int = 1
    saving_threads: int = 1
    # A task that exceeds this is killed (with its worker processes) and recorded as failed.
    task_timeout_sec: float | None = 3600.0

    def verify(self) -> str:
        resolved = shutil.which(self.executable)
        if not resolved:
            candidate = Path(self.executable)
            if candidate.is_file():
                resolved = str(candidate.resolve())
        if not resolved:
            raise SegmentationIntegrationError(
                "TotalSegmentator executable is unavailable; install the pinned local repository"
            )
        self._verify_registry()
        if self.weights_directory is None or not self.weights_directory.is_dir():
            raise SegmentationIntegrationError(
                f"TotalSegmentator weights directory is unavailable: {self.weights_directory}"
            )
        self._verify_weights()
        return resolved

    def _verify_weights(self) -> None:
        assert self.weights_directory is not None
        required = {**TASK_WEIGHT_IDS, "total": (297,) if self.fast else (291, 292, 293, 294, 295)}
        missing = []
        for task, dataset_ids in required.items():
            for dataset_id in dataset_ids:
                candidates = self.weights_directory.glob(f"Dataset{dataset_id}_*")
                complete = any(
                    (model_dir / "plans.json").is_file()
                    and (model_dir / "dataset.json").is_file()
                    and (model_dir / "fold_0" / "checkpoint_final.pth").is_file()
                    and (model_dir / "fold_0" / "checkpoint_final.pth").stat().st_size > 1_000_000
                    for dataset_dir in candidates if dataset_dir.is_dir()
                    for model_dir in dataset_dir.iterdir() if model_dir.is_dir()
                )
                if not complete:
                    missing.append(f"{task}:Dataset{dataset_id}")
        if missing:
            raise SegmentationIntegrationError(
                "TotalSegmentator weights are incomplete: " + ", ".join(missing)
            )

    def _verify_registry(self) -> None:
        if self.repository is None or not self.repository.is_dir():
            raise SegmentationIntegrationError(f"TotalSegmentator repository is unavailable: {self.repository}")
        sys.path.insert(0, str(self.repository.resolve()))
        try:
            from totalsegmentator.registry import get_task_classes

            for task, required in TASK_CLASSES.items():
                available = set(get_task_classes(task).values())
                missing = set(required) - available
                if missing:
                    raise SegmentationIntegrationError(
                        f"pinned TotalSegmentator task {task!r} lacks classes {sorted(missing)}"
                    )
        finally:
            if sys.path and sys.path[0] == str(self.repository.resolve()):
                sys.path.pop(0)

    def _command(self, input_path: Path, output_directory: Path, task: str, device: str) -> list[str]:
        command = [
            self.verify(),
            "-i",
            str(input_path),
            "-o",
            str(output_directory),
            "-ta",
            task,
            "-d",
            device,
            "--report",
            str(output_directory / "run_report.json"),
            "--quiet",
            "--nr_thr_resamp",
            str(max(1, int(self.resample_threads))),
            "--nr_thr_saving",
            str(max(1, int(self.saving_threads))),
        ]
        if task == "total":
            command += ["--roi_subset", *TASK_CLASSES[task]]
        if self.fast and task == "total":
            command.append("--fast")
        return [*command, *self.extra_arguments]

    def _run_task(
        self,
        input_path: Path,
        task_root: Path,
        task: str,
        device: str,
        log: Callable[[str], None] | None,
    ) -> tuple[Path, str | None]:
        destination = task_root / task
        destination.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["TOTALSEG_WEIGHTS_PATH"] = str(self.weights_directory.resolve())
        command = self._command(input_path, destination, task, device)
        if log:
            log(f"TotalSegmentator task={task} command={subprocess.list2cmdline(command)}")
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,  # own process group, so a timeout also kills its workers
        )
        try:
            stdout, stderr = process.communicate(timeout=self.task_timeout_sec)
        except subprocess.TimeoutExpired:
            import signal

            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            message = (
                f"timeout after {self.task_timeout_sec:.0f} s (a worker may have been OOM-killed); "
                "task killed"
            )
            if log:
                log(f"TotalSegmentator task={task} {message}")
            return destination, message
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if log and result.stdout.strip():
            log(f"TotalSegmentator task={task} stdout:\n{result.stdout.strip()}")
        if log and result.stderr.strip():
            log(f"TotalSegmentator task={task} stderr:\n{result.stderr.strip()}")
        if log:
            log(
                f"TotalSegmentator task={task} exit_code={result.returncode} "
                f"elapsed_sec={time.perf_counter() - started:.1f}"
            )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit_code={result.returncode}"
            return destination, detail[-2000:]
        return destination, None

    def run(
        self,
        input_path: Path,
        output_directory: Path,
        *,
        device: str,
        log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run every TotalSegmentator task and write the project masks.

        Masks are written directly as ``output_directory/<anatomy>.nii.gz``. Raw task
        output lives only in a temporary scratch directory for the duration of this call.
        """
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_directory.mkdir(parents=True, exist_ok=True)
        scratch_parent = None
        if self.scratch_directory is not None:
            self.scratch_directory.mkdir(parents=True, exist_ok=True)
            scratch_parent = str(self.scratch_directory)
        with tempfile.TemporaryDirectory(prefix="totalseg_", dir=scratch_parent) as scratch:
            result = self._run_in_scratch(input_path, output_directory, Path(scratch), device, log)
        # A completed study never keeps the old nested layout (tasks/ + canonical/) beside
        # the flat masks. This runs only after every new mask has been written.
        for name in LEGACY_STUDY_DIRECTORIES:
            legacy = output_directory / name
            if legacy.is_dir():
                shutil.rmtree(legacy, ignore_errors=True)
        return result

    def _run_in_scratch(
        self,
        input_path: Path,
        output_directory: Path,
        scratch: Path,
        device: str,
        log: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        task_roots: dict[str, Path] = {}
        errors: dict[str, str] = {}
        for task in TASK_CLASSES:
            task_roots[task], error = self._run_task(input_path, scratch, task, device, log)
            if error:
                errors[task] = error

        # Header only: the masks inherit the CT geometry, the voxels are not needed here.
        reference = load_image(input_path)
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        paths: dict[str, Path] = {}
        mask_provenance: dict[str, dict[str, Any]] = {}

        def source(task: str, name: str) -> Path:
            return task_roots[task] / f"{name}.nii.gz"

        def destination(name: str) -> Path:
            return output_directory / f"{name}.nii.gz"

        def written(name: str, started: float, mask: np.ndarray) -> None:
            if log:
                log(
                    f"mask={name} written voxels={int(np.count_nonzero(mask))} "
                    f"elapsed_sec={time.perf_counter() - started:.1f}"
                )

        def failed(name: str, message: str) -> None:
            errors[name] = message
            if log:
                log(f"mask={name} unavailable reason={message}")

        def record(
            name: str,
            *,
            task: str | list[str],
            source_names: list[str],
            postprocessing: str,
            approximation: bool = False,
            fallback: str | None = None,
            parameters: dict[str, Any] | None = None,
        ) -> None:
            mask_provenance[name] = {
                "source_model": "TotalSegmentator",
                "source_task": task,
                "source_classes": source_names,
                "postprocessing": postprocessing,
                "parameters": dict(parameters or {}),
                "is_approximation": bool(approximation),
                "fallback": fallback,
            }

        def save_one(name: str, task: str, source_name: str) -> None:
            started = time.perf_counter()
            candidate = source(task, source_name)
            if candidate.is_file():
                array, image = load_nifti(candidate)
                if not same_geometry(image, reference):
                    errors[task] = f"output geometry mismatch for {source_name}"
                    return
                mask = array > 0
                paths[name] = save_binary_mask(mask, reference, destination(name))
                record(
                    name,
                    task=task,
                    source_names=[source_name],
                    postprocessing="binarize(value>0)",
                )
                written(name, started, mask)

        def loaded(names: list[tuple[str, str]]) -> list[np.ndarray] | None:
            values: list[np.ndarray] = []
            for task, source_name in names:
                candidate = source(task, source_name)
                if not candidate.is_file():
                    return None
                array, image = load_nifti(candidate)
                if not same_geometry(image, reference):
                    errors[task] = f"output geometry mismatch for {source_name}"
                    return None
                values.append(np.asarray(array, dtype=bool))
            return values

        def save_union(
            name: str,
            names: list[tuple[str, str]],
            *,
            postprocessing: str,
            approximation: bool = False,
            parameters: dict[str, Any] | None = None,
        ) -> bool:
            started = time.perf_counter()
            values = loaded(names)
            if values is None:
                return False
            mask = np.logical_or.reduce(values)
            paths[name] = save_binary_mask(mask, reference, destination(name))
            record(
                name,
                task=sorted({task for task, _ in names}),
                source_names=[source_name for _, source_name in names],
                postprocessing=postprocessing,
                approximation=approximation,
                parameters=parameters,
            )
            written(name, started, mask)
            return True

        lung_parts = [source("total", name) for name in TASK_CLASSES["total"] if name.startswith("lung_")]
        if all(path.is_file() for path in lung_parts):
            started = time.perf_counter()
            loaded_lungs = [load_nifti(path) for path in lung_parts]
            if all(same_geometry(image, reference) for _, image in loaded_lungs):
                lung = combine_masks(array for array, _ in loaded_lungs)
                paths["lung"] = save_binary_mask(lung, reference, destination("lung"))
                record(
                    "lung",
                    task="total",
                    source_names=[path.stem.removesuffix(".nii") for path in lung_parts],
                    postprocessing="binary UNION of five lung lobes",
                )
                written("lung", started, lung)
                # Lobes and sides reuse the arrays already loaded for `lung`: no extra read.
                lobes = {
                    path.stem.removesuffix(".nii"): np.asarray(array) > 0
                    for path, (array, _) in zip(lung_parts, loaded_lungs)
                }
                for name in LUNG_LOBES:
                    started = time.perf_counter()
                    paths[name] = save_binary_mask(lobes[name], reference, destination(name))
                    record(name, task="total", source_names=[name], postprocessing="binarize(value>0)")
                    written(name, started, lobes[name])
                for side, members in LUNG_SIDES.items():
                    started = time.perf_counter()
                    mask = np.logical_or.reduce([lobes[name] for name in members])
                    paths[side] = save_binary_mask(mask, reference, destination(side))
                    record(
                        side,
                        task="total",
                        source_names=list(members),
                        postprocessing=f"binary UNION of the {side.removeprefix('lung_')} lung lobes",
                    )
                    written(side, started, mask)
            else:
                errors["total"] = "output geometry mismatch for one or more lung lobes"
        save_one("heart", "total", "heart")
        save_one("mediastinum", "trunk_cavities", "mediastinum")
        save_one("myocardium", "heartchambers_highres", "heart_myocardium")
        save_one("central_pa", "heartchambers_highres", "pulmonary_artery")
        save_one("lung_arteries", "lung_vessels", "lung_arteries")
        save_one("lung_veins", "lung_vessels", "lung_veins")
        save_one("airways", "lung_vessels", "lung_airways")
        save_one("rv", "heartchambers_highres", "heart_ventricle_right")
        save_one("lv", "heartchambers_highres", "heart_ventricle_left")
        save_one("ra", "heartchambers_highres", "heart_atrium_right")
        save_one("la", "heartchambers_highres", "heart_atrium_left")
        chamber_sources = [
            ("heartchambers_highres", name)
            for name in (
                "heart_myocardium",
                "heart_ventricle_right",
                "heart_ventricle_left",
                "heart_atrium_right",
                "heart_atrium_left",
            )
        ]
        if not save_union(
            "strict_heart",
            chamber_sources,
            postprocessing="binary UNION of myocardium and four cardiac chambers; excludes PA/aorta classes",
        ) and "heart" in paths:
            started = time.perf_counter()
            heart, _ = load_nifti(paths["heart"])
            strict_heart = heart > 0
            paths["strict_heart"] = save_binary_mask(strict_heart, reference, destination("strict_heart"))
            record(
                "strict_heart",
                task="total",
                source_names=["heart"],
                postprocessing="binarize generic heart fallback",
                approximation=True,
                fallback="generic heart used because complete high-resolution chamber set was unavailable",
            )
            written("strict_heart", started, strict_heart)
        save_union(
            "lung_vessels",
            [("lung_vessels", "lung_arteries"), ("lung_vessels", "lung_veins")],
            postprocessing="binary UNION of segmented intrapulmonary arteries and veins",
            approximation=True,
        )
        save_union(
            "pa_tree",
            [("heartchambers_highres", "pulmonary_artery"), ("lung_vessels", "lung_arteries")],
            postprocessing="central pulmonary artery UNION intrapulmonary artery segmentation",
            approximation=True,
        )
        save_union(
            "body",
            [("body", "body_trunc"), ("body", "body_extremities")],
            postprocessing="binary UNION of body trunk and extremities",
        )

        # Derived shells/neighbourhoods use an exact physical-distance transform processed in
        # z-slabs. scipy binary_erosion/dilation with a 12-15 mm ball would allocate a
        # 5-15 GB offset table on a 512x512 CTPA and is O(voxels x ball size).
        if "body" in paths:
            started = time.perf_counter()
            try:
                body, _ = load_nifti(paths["body"])
                wall = boundary_shell(body > 0, spacing, self.body_wall_thickness_mm)
                paths["body_wall"] = save_binary_mask(wall, reference, destination("body_wall"))
                record(
                    "body_wall",
                    task="derived",
                    source_names=["body"],
                    postprocessing="body voxels within wall_thickness_mm of the body surface (exact EDT)",
                    approximation=True,
                    parameters={"wall_thickness_mm": self.body_wall_thickness_mm},
                )
                written("body_wall", started, wall)
            except (ModuleNotFoundError, ValueError, MemoryError) as exc:
                failed("body_wall", f"{type(exc).__name__}: {exc}")

        if all(name in paths for name in ("central_pa", "lung_vessels", "mediastinum")):
            started = time.perf_counter()
            try:
                central_pa, _ = load_nifti(paths["central_pa"])
                lung_vessels, _ = load_nifti(paths["lung_vessels"])
                mediastinum, _ = load_nifti(paths["mediastinum"])
                neighbourhood = physical_dilation(
                    (mediastinum > 0) | (central_pa > 0), spacing, self.hilar_proximity_mm
                )
                hilar = (central_pa > 0) | ((lung_vessels > 0) & neighbourhood)
                paths["hilar_vessels"] = save_binary_mask(hilar, reference, destination("hilar_vessels"))
                record(
                    "hilar_vessels",
                    task="derived",
                    source_names=["central_pa", "lung_vessels", "mediastinum"],
                    postprocessing=(
                        "central_pa UNION (lung_vessels within proximity_mm of "
                        "mediastinum/central_pa; exact EDT)"
                    ),
                    approximation=True,
                    parameters={"proximity_mm": self.hilar_proximity_mm},
                )
                written("hilar_vessels", started, hilar)
            except (ModuleNotFoundError, ValueError, MemoryError) as exc:
                failed("hilar_vessels", f"{type(exc).__name__}: {exc}")
        provenance = {
            "backend": "totalsegmentator",
            "tasks": {task: list(classes) for task, classes in TASK_CLASSES.items()},
            "task_errors": errors,
            "masks": mask_provenance,
            "scientific_scope": {
                "central_pa": "TotalSegmentator pulmonary_artery class; approximate central-PA mask, not ground truth",
                "pa_tree": (
                    "union of central and intrapulmonary artery models; boundaries may be discontinuous. "
                    "When lung_arteries already contains the central PA the union equals lung_arteries "
                    "(qc_summary.csv duplicate_of)"
                ),
                "hilar_vessels": "proximity-derived approximation; not a validated hilar-vessel annotation",
                "body_wall": "morphological shell used only as a negative-control candidate region",
            },
        }
        return {
            "masks": paths,
            "errors": errors,
            "provenance": provenance,
            "mask_provenance": mask_provenance,
        }
