"""The three baseline experiments: which models x heads x training fractions each one needs.

The grids themselves are data: scripts/diagnosis/baselines/<experiment>/experiment.yaml (models, heads,
fractions, optional per-model `overrides`). A model with overrides in an experiment gets its
own runs, tagged ``__v<experiment>``, so it never collides with the shared runs below.

A run is identified only by its scientific settings (model, head, fraction, fold, seed), not
by the experiment that asked for it, so ResNet-18 3D + MLP on 100% of the data is trained
once and shared by exp01 (baselines), exp02 (its 100% point) and exp03 (its MLP arm):

    <outputs>/diagnosis/BASE/<profile>/<task>/runs/<model>__<head>__frac<PPP>/<fold>_seed<S>/epoch_<E>/
    <outputs>/diagnosis/BASE/<profile>/<task>/<experiment>/   summary tables, plots, runs.csv

A case run with training settings that differ from its config (--scratch, --lr, --batch-size,
--accumulation, --patience, --no-epoch-auc, extra --set) gets a ``__x<settings>`` suffix
(settings_stamp), so it never lands in, or is skipped because of, the default run's folder.
Runs with the config's own settings keep the unsuffixed name.
"""
from __future__ import annotations

MODEL_GROUPS = {
    "resnet18_2d": "CNN", "resnet18_25d": "CNN", "resnet18_3d": "CNN", "resnet50_3d": "CNN",
    "densenet121_3d": "CNN", "convnext_2d": "CNN", "convnext_25d": "CNN", "convnext_3d": "CNN",
    "vit_2d": "Transformer", "vit_25d": "Transformer", "vit_3d": "Transformer",
    "swin_2d": "Transformer", "swin_25d": "Transformer", "swin_3d": "Transformer",
    "nnmamba_3d": "Mamba", "mamba_mae_3d": "Mamba", "vmamba_3d": "Mamba",
    "penet_3d": "PE-specific", "ctfm_lora_3d": "PE-specific / FM", "ctfm_frozen_3d": "PE-specific / FM",
}

EXPERIMENT_NAMES = ("exp01_baselines", "exp02_data_fraction", "exp03_head_ablation")


def _load_experiments() -> dict[str, dict]:
    """Each experiment's grid lives in scripts/diagnosis/baselines/<experiment>/experiment.yaml."""
    from pathlib import Path

    import yaml

    scripts = Path(__file__).resolve().parents[2] / "scripts" / "diagnosis" / "baselines"
    experiments = {}
    for name in EXPERIMENT_NAMES:
        spec = yaml.safe_load((scripts / name / "experiment.yaml").read_text(encoding="utf-8")) or {}
        missing = [key for key in ("title", "models", "heads", "fractions") if key not in spec]
        if missing:
            raise ValueError(f"scripts/diagnosis/baselines/{name}/experiment.yaml is missing {missing}")
        spec["fractions"] = [int(value) for value in spec["fractions"]]
        spec["overrides"] = {str(key): list(value or []) for key, value in (spec.get("overrides") or {}).items()}
        unknown = sorted(set(spec["overrides"]) - set(spec["models"]))
        if unknown:
            raise ValueError(f"scripts/diagnosis/baselines/{name}/experiment.yaml overrides unknown models {unknown}")
        experiments[name] = spec
    return experiments


EXPERIMENTS = _load_experiments()


def task_directory(task: str, cohort: str = "all", target: str = "1_month_mortality") -> str:
    return "diagnosis" if task == "diagnosis" else f"prognosis_{cohort}_{target}"


def outputs_root():
    """The persistent output root (configs/paths.yaml + PE_* environment overrides)."""
    from pathlib import Path

    import yaml

    from source.data.paths import ProjectPaths
    from source.utils.config import expand_environment

    text = (Path(__file__).resolve().parents[2] / "configs" / "paths.yaml").read_text(encoding="utf-8")
    return ProjectPaths.resolve(expand_environment(yaml.safe_load(text))).output_root


def base_directory(profile: str, task_dir: str):
    family = "diagnosis" if task_dir == "diagnosis" else "prognosis"
    return outputs_root() / family / "BASE" / profile / task_dir


CONFIG_ROOT_PARTS = ("configs", "runs", "02_diagnosis", "baselines")
DIMENSION_FOLDERS = {"2D": "2D", "2.5D": "2_5D", "3D": "3D"}


def config_path(model: str):
    """configs/runs/02_diagnosis/baselines/<2D|2_5D|3D>/<model>.yaml"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2].joinpath(*CONFIG_ROOT_PARTS)
    path = root / DIMENSION_FOLDERS[model_dimension(model)] / f"{model}.yaml"
    if not path.is_file():
        raise SystemExit(f"unknown baseline model {model!r}: no {path}")
    return path


def dimension_folder(model: str) -> str:
    return DIMENSION_FOLDERS[model_dimension(model)]


def model_dimension(model: str) -> str:
    return "2.5D" if model.endswith("_25d") else model.rsplit("_", 1)[-1].upper()


def run_tag(model: str, head: str, fraction: int, variant: str = "", settings: str = "") -> str:
    return (f"{model}__{head}__frac{int(fraction):03d}" + (f"__v{variant}" if variant else "")
            + (f"__x{settings}" if settings else ""))


# Overrides that only say where things are or which hardware runs them; they never change
# what a run computes, so they do not split a case into a separate folder.
NON_SCIENTIFIC_PREFIXES = ("paths.", "compute.", "data.profile=")


def settings_stamp(
    config: dict,
    *,
    scratch: bool = False,
    lr: float | None = None,
    batch_size: int | None = None,
    accumulation: int | None = None,
    patience: int | None = None,
    no_epoch_auc: bool = False,
    extra_sets: list[str] | tuple[str, ...] = (),
) -> str:
    """Short, filesystem-safe record of the settings in which a case deviates from ``config``.

    ``config`` is the resolved run config *before* the case's own training flags are applied;
    a flag equal to the configured value is not a deviation. Parts, joined by ``_``:
    ``scratch``, ``lr<g>``, ``bs<N>``, ``acc<N>``, ``pat<N>``, ``noauc``, ``set<sha1[:8]>``.
    """
    import hashlib

    training = dict(config.get("training") or {})
    pretrained = dict((config.get("model") or {}).get("pretrained") or {})
    parts: list[str] = []
    if scratch and pretrained.get("enabled", True) is not False:
        parts.append("scratch")
    if lr is not None and float(lr) != float(training.get("learning_rate") or 0.0):
        parts.append(f"lr{float(lr):g}")
    for prefix, value, key in (("bs", batch_size, "batch_size"), ("acc", accumulation, "gradient_accumulation"),
                               ("pat", patience, "early_stopping_patience")):
        if value is not None and int(value) != int(training.get(key) or 0):
            parts.append(f"{prefix}{int(value)}")
    if no_epoch_auc and training.get("record_epoch_auc", True) is not False:
        parts.append("noauc")
    scientific = sorted(str(item) for item in extra_sets if not str(item).startswith(NON_SCIENTIFIC_PREFIXES))
    if scientific:
        parts.append("set" + hashlib.sha1("\n".join(scientific).encode("utf-8")).hexdigest()[:8])
    return "_".join(parts)
