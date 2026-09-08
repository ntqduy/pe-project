# `tools/` — CLI reference

Normally you do not call these directly: `python run.py <command> <experiment>` resolves a
semantic name from `configs/experiments.yaml` and delegates here. This file is for when you
want the underlying command, a nested entrypoint, or an option `run.py` does not pass through.

Run everything from the project root. `tools/` holds entrypoints and orchestration only; model
and pipeline logic lives in `source/`.

```text
tools/
├── _common.py              parser, config, dataset, patient filter, atomic table writers
├── preflight.py            pre-run validation; never trains or infers
├── launch.py               one training or generation job on CPU / one GPU / DDP
├── launch_parallel.py      several independent jobs across GPU groups, in waves
├── build_summary.py        aggregate existing result.json files
├── build_experiment_map.py regenerate docs/EXPERIMENT_MAP.md from the registry
├── sync_project.py         copy source/configs/scripts to the cloud project root
├── create_masks/generate_masks.py    TotalSegmentator + LungMask QC
├── build_rois/build_rois.py          ROI1-ROI8 from a stored segmentation run
├── silver_labels/generate_silver_labels.py
├── pretrain_model/{materialize_foundation,train_dapt,train_alignment,train_silver_encoder}.py
├── tasks/{train_task,evaluate,counterfactual}.py
└── data/{build_dataset,create_split}.py
```

## `data/build_dataset.py`

Builds one dataset profile from the read-only INSPECT release. It holds no scientific logic:
it resolves the run config, loads the profile it names from `source/dataset/profiles/`, and
calls `source/data_preprocessing/pipeline.build_dataset`, which is the single implementation
all profiles share.

```bash
python run.py run data.dataset.smoke_30 --allow-full            # technical smoke
python run.py run data.dataset.test_500_sample --allow-full     # the whole 500-patient subset
python run.py run data.dataset.full_inspect    --allow-full     # the whole cohort
```

Extra flags this CLI accepts that `run.py` does not forward:

| flag | effect |
|---|---|
| `--no-preprocess` | write manifests pointing at the raw read-only volumes, skipping the cache |
| `--metadata-only` | skip file-existence rules; validate the cohort from metadata alone |

Stages inside `source/data_preprocessing/`, in order: `sources` (join the official tables),
`filters` (eligibility, noise removal, exclusion ledger), `integrity` (corrupted/missing CT),
`adjudication` (per-patient label reconciliation), `sampling` (patient-level, inside the
official split), `leakage` (split preservation), then `volumes`: reorientation, optional
physical resampling, body crop, full-FOV fitting, provenance sidecars and cache QC. The
derived NPY remains compatible with current dense-model loaders; its sidecar and optional
patch grid are the audited geometry record.

## `preflight.py`

Resolves the config and checks data paths, patient split, required files, upstream
models/weights, adapter contracts, supervision artifacts, GPUs, and output collisions. It does
not run a model.

```bash
python tools/preflight.py --config configs/runs/00_data/segmentation.yaml --gpus 0
python tools/preflight.py --config configs/runs/03_diagnosis/anatomy/single_concat.yaml --gpus 0,1
```

`launch.py` and the domain CLIs call preflight themselves, so calling it separately is only
for fixing configuration first. A config that parses is not the same as an experiment that is
ready to run.

## `launch.py`: one job

The launcher reads `experiment.stage` and picks the single matching entrypoint:

| Stage | Entrypoint |
|---|---|
| `dataset` | `data/build_dataset.py` (own CLI, not via `launch.py`) |
| `foundation` | `pretrain_model/materialize_foundation.py` |
| `dapt` | `pretrain_model/train_dapt.py` |
| `alignment` | `pretrain_model/train_alignment.py` |
| `silver` | `silver_labels/generate_silver_labels.py` |
| `silver_encoder_adaptation` | `pretrain_model/train_silver_encoder.py` |
| `diagnosis`, `prognosis`, `contour`, `roi_student` | `tasks/train_task.py` |
| `counterfactual` | `tasks/counterfactual.py` (frozen inference only) |

```bash
python tools/launch.py --config configs/runs/02_representation/dapt/dino.yaml --gpus 0
python tools/launch.py --config configs/runs/02_representation/dapt/dino.yaml --gpus 0,1,2,3
python tools/launch.py --config configs/runs/00_data/silver/hybrid.yaml --gpus 0,1 --allow-full
python tools/launch.py --config configs/runs/02_representation/dapt/dino.yaml --gpus 0,1 --dry-run
```

`--gpus` takes physical IDs. The launcher sets `CUDA_VISIBLE_DEVICES`, remaps the child process
to logical `0..N-1`, and uses `torchrun` from two GPUs up. Foundation materialization is CPU or
one GPU only.

Silver generation is not DDP model training: each rank loads its providers on its local GPU and
processes a shard of reports, then rank 0 merges. SL02 loads both Falcon and MedGemma per rank.

## Data selection

Segmentation, ROI, silver generation and counterfactual inference each require exactly one
selection mode:

| Mode | Segmentation / ROI / counterfactual | Silver generation |
|---|---|---|
| one or more patients | `--patient-id ID` (repeatable) | `--patient-id ID` (repeatable) |
| first N items | `--max-cases N` | `--max-reports N` |
| everything | `--allow-full` | `--allow-full` |

```bash
python tools/create_masks/generate_masks.py --config configs/runs/00_data/segmentation.yaml \
  --patient-id P001 --gpus 0
python tools/build_rois/build_rois.py --config configs/runs/00_data/roi.yaml --patient-id P001
python tools/launch.py --config configs/runs/00_data/silver/hybrid.yaml --patient-id P001 --gpus 0
```

`--patient-id` selects every record for that patient, not one study. For ROI the patient must
exist in the stored segmentation manifest; for silver it must exist in the report table.

## Task tools

`tasks/counterfactual.py` is the necessity stage (`anatomy.remove_*`). It loads one full-model
checkpoint, freezes every parameter, computes the original probability, erases the ROI with the
configured neutral/local-mean policy, and re-runs **the same model with the same original
masks**. It writes `counterfactual_predictions.parquet`, paired bootstrap metrics and
`result.json`. It never calls segmentation and has no optimizer.

`tasks/evaluate.py` also accepts `--patient-id` or `--max-cases` for smoke inference on a real
checkpoint. Smoke artifacts go to `RUN/smoke/<scope-hash>/` and never overwrite the full
`predictions.parquet` / `result.json`.

ROI students and distilled students use `tasks/train_task.py`. A KD run additionally writes
`distillation_validation_predictions.parquet` with teacher and student logits and the GT / KD /
total loss components. The teacher is always frozen, in eval mode, and outside the optimizer.

Prognosis evaluation writes `calibration_curve.parquet`, and `result.json` carries calibration
slope/intercept, Brier score and patient-bootstrap confidence intervals.
`--reference-predictions` runs a paired patient bootstrap and refuses mismatched
patient/study sets or targets.

## `launch_parallel.py`: several jobs

Not DAPT-specific: it accepts any config `launch.py` supports and splits the GPU list into
disjoint groups.

The batch file is yours to write — there is none in this repository — and
`parallel.enabled: true` is an explicit opt-in the launcher refuses to run without:

```yaml
parallel:
  enabled: true
  devices: [0, 1, 2, 3]
  gpus_per_job: 1
  jobs:
    - config: configs/runs/03_diagnosis/anatomy/single_concat.yaml
    - config: configs/runs/03_diagnosis/baseline/global_single.yaml
    - config: configs/runs/04_prognosis/modality/image_ehr_pesi.yaml
```

```bash
python tools/launch_parallel.py --config sweep.yaml --dry-run
python tools/launch_parallel.py --config sweep.yaml
```

Per-job `args` accept only `--patient-id`, `--max-reports`, `--allow-full` and `--resume`;
scientific changes belong in a run config. **`--set` is not accepted per job**, so one
parallel sweep cannot vary `encoder.init_source` or `data.profile` across jobs — every job
must name a distinct config. Run an encoder-initialization comparison sequentially through
`scripts/` instead, or give each cell its own run config. Do not schedule two stages in one
wave when one waits on the other's checkpoint.

## Direct nested entrypoints

`launch.py` is the normal path, but a single process can be debugged directly:

```bash
python tools/pretrain_model/train_dapt.py --config configs/runs/02_representation/dapt/dino.yaml --gpus 0
python tools/pretrain_model/train_alignment.py --config configs/runs/02_representation/image_report_alignment.yaml --gpus 0
python tools/tasks/train_task.py --config configs/runs/03_diagnosis/anatomy/single_concat.yaml --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/anatomy/single_concat.yaml --gpus 0
```

Calling a nested file directly with several GPUs does not spawn several processes; use
`launch.py` for DDP.

## Overrides, resume, overwrite

```bash
python tools/launch.py --config configs/runs/02_representation/silver_adaptation/hybrid.yaml \
  --set silver_training.silver_source=SL01 --gpus 0,1
```

- `--set KEY=VALUE` is YAML-parsed and repeatable.
- `--resume` only where the entrypoint supports resuming.
- `--overwrite` deliberately replaces a run with the same experiment id.
- Data-generation tools keep per-study/per-report state and continue a partial run by default;
  `--overwrite` clears that state.

## Other utilities

```bash
# create one explicit patient-level split
python tools/data/create_split.py --input SOURCE.csv --output TARGET.csv --create-split --seed 42

# aggregate existing result.json files
python tools/build_summary.py

# review the sync plan before executing it
python tools/sync_project.py --help
```
