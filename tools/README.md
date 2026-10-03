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
├── run_status.py           how far a task run got (absent/incomplete/trained/evaluated/different)
├── create_masks/generate_masks.py    TotalSegmentator + LungMask QC
├── build_rois/build_rois.py          ROI1-ROI8 from a stored segmentation run
├── silver_labels/generate_silver_labels.py
├── tasks/{train_task,evaluate}.py
├── tasks/{zeroshot_penet,zeroshot_radar,zeroshot_radar_worker}.py
├── data/{build_dataset,build_ctfm_cache,build_split_manifests}.py
└── baselines/{run_case,run_many,summarize,smoke,smoke_data,smoke_pipeline,fraction_subsets,prepare_weights,experiments}.py
```

## `data/build_dataset.py`

Builds one dataset profile from the read-only INSPECT release. It holds no scientific logic:
it resolves the run config, loads the profile it names from `source/data/profiles/`, and
calls `source/data/build/pipeline.build_dataset`, which is the single implementation
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

Stages inside `source/data/build/`, in order: `sources` (join the official tables),
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
python tools/preflight.py --config configs/runs/00_data/segmentation/inspect.yaml --gpus 0
python tools/preflight.py --config configs/runs/02_diagnosis/baselines/3D/resnet18_3d.yaml --gpus 0,1
```

`launch.py` and the domain CLIs call preflight themselves, so calling it separately is only
for fixing configuration first. A config that parses is not the same as an experiment that is
ready to run.

## `launch.py`: one job

The launcher reads `experiment.stage` and picks the single matching entrypoint:

| Stage | Entrypoint |
|---|---|
| `dataset` | `data/build_dataset.py` (own CLI, not via `launch.py`) |
| `silver` | `silver_labels/generate_silver_labels.py` |
| `diagnosis`, `prognosis` | `tasks/train_task.py` (`--evaluate` → `tasks/evaluate.py`) |

```bash
python tools/launch.py --config configs/runs/02_diagnosis/baselines/3D/resnet18_3d.yaml --gpus 0
python tools/launch.py --config configs/runs/02_diagnosis/baselines/3D/resnet18_3d.yaml --gpus 0,1,2,3
python tools/launch.py --config configs/runs/00_data/silver/medgemma.yaml --gpus 0,1 --allow-full
python tools/launch.py --config configs/runs/02_diagnosis/baselines/3D/resnet18_3d.yaml --gpus 0,1 --dry-run
```

`--gpus` takes physical IDs. The launcher sets `CUDA_VISIBLE_DEVICES`, remaps the child process
to logical `0..N-1`, and uses `torchrun` from two GPUs up.

Silver generation is not DDP model training: each rank loads its providers on its local GPU and
processes a shard of reports, then rank 0 merges. the medgemma method loads MedGemma once per rank.

## Data selection

Segmentation, ROI and silver generation each require exactly one selection mode:

| Mode | Segmentation / ROI | Silver generation |
|---|---|---|
| one or more patients | `--patient-id ID` (repeatable) | `--patient-id ID` (repeatable) |
| first N items | `--max-cases N` | `--max-reports N` |
| everything | `--allow-full` | `--allow-full` |

```bash
python tools/create_masks/generate_masks.py --config configs/runs/00_data/segmentation/inspect.yaml \
  --patient-id P001 --gpus 0
python tools/build_rois/build_rois.py --config configs/runs/00_data/roi/inspect.yaml --patient-id P001
python tools/launch.py --config configs/runs/00_data/silver/medgemma.yaml --patient-id P001 --gpus 0
```

`--patient-id` selects every record for that patient, not one study. For ROI the patient must
exist in the stored segmentation manifest; for silver it must exist in the report table.

## Task tools

`tasks/evaluate.py` also accepts `--patient-id` or `--max-cases` for smoke inference on a real
checkpoint. Smoke artifacts (`result.csv`, `predictions.csv`, `logs.txt`, `result.json`) go to
`RUN/smoke/<scope-hash>/` and never overwrite the full-test files in `RUN/epoch_<N>/` and
`RUN/result.json`.

`tasks/evaluate.py --restrict-to <csv>` limits evaluation to one shared list of
`patient_id[,study_id]`, so every arm of a comparison is scored on identical cases.

`tasks/evaluate.py` also writes the Grad-CAM preview (`preview/`, or `visualize/{correct,incorrect}/`
for the baseline zoo's `preview.split: test`); re-running evaluate rebuilds it. Details:
`docs/03_training_evaluation.md`.

Prognosis evaluation writes `calibration_curve.parquet`, and `result.json` carries calibration
slope/intercept, Brier score and patient-bootstrap confidence intervals.
`--reference-predictions` runs a paired patient bootstrap (generic comparison of two models on
the same test patients) and refuses mismatched patient/study sets or targets.

## `baselines/`: the baseline experiments

`run_case.py` runs one case (prepare split manifests → train → evaluate), `run_many.py` a whole
experiment grid in parallel over GPU slots, `summarize.py` the per-experiment tables
(`summary.md`, `summary_pretty.csv`, `summary_ensemble.csv`, `summary.csv`, `summary_raw.csv`).
`experiments.py` reads `scripts/diagnosis/baselines/<exp>/experiment.yaml`; `fraction_subsets.py`
exports and checks the exp02 training subsets per seed (`--dir <seed dir>` re-checks one);
`prepare_weights.py` fetches weights once; `smoke.py` runs one real train step per arm;
`smoke_data.py` writes a tiny synthetic dataset and `smoke_pipeline.py` runs the whole pipeline
on it (CPU is enough; `output/_smoke` or `$PE_SMOKE_ROOT`). Usually driven by
`scripts/tool/run_baseline_grid.sh`; see `scripts/diagnosis/baselines/README.md`.

```bash
python tools/baselines/run_case.py --model resnet18_3d --head kan --fraction 50 --seed 1 --gpus 0
python tools/baselines/run_many.py --exp exp01_baselines --gpus 0,1,2,3
python tools/baselines/smoke_pipeline.py
```

## Direct nested entrypoints

`launch.py` is the normal path, but a single process can be debugged directly:

```bash
python tools/tasks/train_task.py --config configs/runs/02_diagnosis/baselines/3D/resnet18_3d.yaml --gpus 0
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/baselines/3D/resnet18_3d.yaml --gpus 0
```

Calling a nested file directly with several GPUs does not spawn several processes; use
`launch.py` for DDP.

## Overrides, resume, overwrite

```bash
python tools/launch.py --config configs/runs/02_diagnosis/baselines/3D/resnet18_3d.yaml \
  --set training.epochs=1 --gpus 0,1
```

- `--set KEY=VALUE` is YAML-parsed and repeatable.
- `--resume` only where the entrypoint supports resuming.
- `--overwrite` deliberately replaces a run with the same experiment id.
- Data-generation tools keep per-study/per-report state and continue a partial run by default;
  `--overwrite` clears that state. Segmentation and ROI refuse to resume when an
  output-affecting setting differs from the one recorded in the run's `resume_settings.json`
  (the error names the changed keys); the silver cache instead re-labels any report whose
  generator signature (`RULE_VERSION`, thresholds, provider kwargs, prompt hash) changed.

## Other utilities

```bash
# training-fraction manifests (exp02) of the official split: train subsampled, validation/test unchanged
python tools/data/build_split_manifests.py --profile full_inspect \
    --base-manifest manifests/diagnosis.csv --label pe_present --seed 0 \
    --apply manifests/diagnosis.csv --fraction 25 50 75 100
#   --fraction: a percent when it ends in % or is > 1 (25, 12.5, 1%), a fraction when < 1
#   (0.25); a bare 1 is rejected. Tags are lossless (12.5% -> frac012p5).
#   tools/baselines/run_case.py --fraction is always a whole percent (1 = 1%).

# CT-FM feature cache for ctfm_frozen_3d (normally via scripts/diagnosis/baselines/prepare_ctfm_cache.sh)
python tools/data/build_ctfm_cache.py --dataset-root <derived>/datasets/<profile> \
    --raw-root <raw INSPECT> --output-name ct_fm --device cuda:0

# how far a run got, for the same --config/--set as launch.py (--format text|tsv|json; tsv/json for scripts)
python tools/run_status.py --config configs/runs/02_diagnosis/baselines/3D/ctfm_frozen_3d.yaml --format json
```

No tool creates a train/validation/test split: the official INSPECT split comes from the
dataset build (`data/build_dataset.py`).
