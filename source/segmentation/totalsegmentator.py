from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from source.imaging.nifti import combine_masks, load_nifti, same_geometry, save_binary_mask


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


@dataclass(frozen=True)
class TotalSegmentatorRunner:
    executable: str = "TotalSegmentator"
    repository: Path | None = None
    weights_directory: Path | None = None
    fast: bool = False
    body_wall_thickness_mm: float = 15.0
    hilar_proximity_mm: float = 12.0
    extra_arguments: tuple[str, ...] = ()

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
        return resolved

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
        ]
        if task == "total":
            command += ["--roi_subset", *TASK_CLASSES[task]]
        if self.fast and task == "total":
            command.append("--fast")
        return [*command, *self.extra_arguments]

    def _run_task(
        self,
        input_path: Path,
        root: Path,
        task: str,
        device: str,
        log: Callable[[str], None] | None,
    ) -> tuple[Path, str | None]:
        destination = root / "tasks" / task
        destination.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["TOTALSEG_WEIGHTS_PATH"] = str(self.weights_directory.resolve())
        command = self._command(input_path, destination, task, device)
        if log:
            log(f"TotalSegmentator task={task} command={subprocess.list2cmdline(command)}")
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        if log and result.stdout.strip():
            log(f"TotalSegmentator task={task} stdout:\n{result.stdout.strip()}")
        if log and result.stderr.strip():
            log(f"TotalSegmentator task={task} stderr:\n{result.stderr.strip()}")
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
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_directory.mkdir(parents=True, exist_ok=True)
        task_roots: dict[str, Path] = {}
        errors: dict[str, str] = {}
        for task in TASK_CLASSES:
            task_roots[task], error = self._run_task(
                input_path, output_directory, task, device, log
            )
            if error:
                errors[task] = error

        _, reference = load_nifti(input_path)
        canonical = output_directory / "canonical"
        paths: dict[str, Path] = {}
        mask_provenance: dict[str, dict[str, Any]] = {}

        def source(task: str, name: str) -> Path:
            return task_roots[task] / f"{name}.nii.gz"

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
            candidate = source(task, source_name)
            if candidate.is_file():
                array, image = load_nifti(candidate)
                if not same_geometry(image, reference):
                    errors[task] = f"output geometry mismatch for {source_name}"
                    return
                paths[name] = save_binary_mask(array > 0, reference, canonical / f"{name}.nii.gz")
                record(
                    name,
                    task=task,
                    source_names=[source_name],
                    postprocessing="binarize(value>0)",
                )

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
            values = loaded(names)
            if values is None:
                return False
            paths[name] = save_binary_mask(
                np.logical_or.reduce(values), reference, canonical / f"{name}.nii.gz"
            )
            record(
                name,
                task=sorted({task for task, _ in names}),
                source_names=[source_name for _, source_name in names],
                postprocessing=postprocessing,
                approximation=approximation,
                parameters=parameters,
            )
            return True

        def ellipsoid(radius_mm: float) -> np.ndarray:
            spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
            radii = np.maximum(1, np.ceil(float(radius_mm) / spacing).astype(int))
            grids = np.ogrid[tuple(slice(-int(value), int(value) + 1) for value in radii)]
            distance = sum((grid * spacing[index] / float(radius_mm)) ** 2 for index, grid in enumerate(grids))
            return np.asarray(distance <= 1.0, dtype=bool)

        lung_parts = [source("total", name) for name in TASK_CLASSES["total"] if name.startswith("lung_")]
        if all(path.is_file() for path in lung_parts):
            loaded_lungs = [load_nifti(path) for path in lung_parts]
            if all(same_geometry(image, reference) for _, image in loaded_lungs):
                paths["lung"] = save_binary_mask(
                    combine_masks(array for array, _ in loaded_lungs), reference, canonical / "lung.nii.gz"
                )
                record(
                    "lung",
                    task="total",
                    source_names=[path.stem.removesuffix(".nii") for path in lung_parts],
                    postprocessing="binary UNION of five lung lobes",
                )
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
            heart, _ = load_nifti(paths["heart"])
            paths["strict_heart"] = save_binary_mask(
                heart > 0, reference, canonical / "strict_heart.nii.gz"
            )
            record(
                "strict_heart",
                task="total",
                source_names=["heart"],
                postprocessing="binarize generic heart fallback",
                approximation=True,
                fallback="generic heart used because complete high-resolution chamber set was unavailable",
            )
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

        if "body" in paths:
            try:
                from scipy import ndimage

                body, _ = load_nifti(paths["body"])
                structure = ellipsoid(self.body_wall_thickness_mm)
                eroded = ndimage.binary_erosion(body.astype(bool), structure=structure, border_value=0)
                wall = body.astype(bool) & ~eroded
                paths["body_wall"] = save_binary_mask(
                    wall, reference, canonical / "body_wall.nii.gz"
                )
                record(
                    "body_wall",
                    task="derived",
                    source_names=["body"],
                    postprocessing="body MINUS physical-space binary erosion",
                    approximation=True,
                    parameters={"wall_thickness_mm": self.body_wall_thickness_mm},
                )
            except (ModuleNotFoundError, ValueError) as exc:
                errors["body_wall"] = f"{type(exc).__name__}: {exc}"

        if all(name in paths for name in ("central_pa", "lung_vessels", "mediastinum")):
            try:
                from scipy import ndimage

                central_pa, _ = load_nifti(paths["central_pa"])
                lung_vessels, _ = load_nifti(paths["lung_vessels"])
                mediastinum, _ = load_nifti(paths["mediastinum"])
                neighbourhood = ndimage.binary_dilation(
                    mediastinum.astype(bool) | central_pa.astype(bool),
                    structure=ellipsoid(self.hilar_proximity_mm),
                )
                hilar = central_pa.astype(bool) | (lung_vessels.astype(bool) & neighbourhood)
                paths["hilar_vessels"] = save_binary_mask(
                    hilar, reference, canonical / "hilar_vessels.nii.gz"
                )
                record(
                    "hilar_vessels",
                    task="derived",
                    source_names=["central_pa", "lung_vessels", "mediastinum"],
                    postprocessing=(
                        "central_pa UNION (lung_vessels INTERSECT physical-space dilation of "
                        "mediastinum/central_pa)"
                    ),
                    approximation=True,
                    parameters={"proximity_mm": self.hilar_proximity_mm},
                )
            except (ModuleNotFoundError, ValueError) as exc:
                errors["hilar_vessels"] = f"{type(exc).__name__}: {exc}"
        provenance = {
            "backend": "totalsegmentator",
            "tasks": {task: list(classes) for task, classes in TASK_CLASSES.items()},
            "task_errors": errors,
            "masks": mask_provenance,
            "scientific_scope": {
                "central_pa": "TotalSegmentator pulmonary_artery class; approximate central-PA mask, not ground truth",
                "pa_tree": "union of central and intrapulmonary artery models; boundaries may be discontinuous",
                "hilar_vessels": "proximity-derived approximation; not a validated hilar-vessel annotation",
                "body_wall": "morphological shell used only as a negative-control candidate region",
            },
        }
        # Raw task outputs are only intermediates used to build the canonical masks above.
        # Their provenance is already embedded in every manifest row, so retaining them
        # would roughly duplicate the per-study mask tree and make review unnecessarily hard.
        shutil.rmtree(output_directory / "tasks", ignore_errors=True)
        return {
            "masks": paths,
            "errors": errors,
            "provenance": provenance,
            "mask_provenance": mask_provenance,
        }
