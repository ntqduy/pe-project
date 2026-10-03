# PE Project — 3D CTPA

Research framework for pulmonary embolism (PE) tasks on 3D CTPA (Stanford INSPECT).
Every stage has its own config, checkpoint lineage, QC and outputs. Only real models and real
pipelines: nothing is mocked, and unresolved data contracts fail loudly instead of being
guessed.

## Where this repository is right now

`python run.py list` is the authoritative experiment inventory (30 entries); blocked entries
state the missing artifact or unresolved contract instead of guessing.

| | state |
|---|---|
| Raw Stanford INSPECT release | present, read-only. 23,248 studies / 19,405 patients |
| Dataset profiles | code and configs complete, **not built yet** — build one first |
| MedGemma (silver labels) | weights staged locally, id and revision recorded |
| CT-FM | staged: strict load of the HF feature extractor (see `third_party/versions.yaml`) |
| Baseline zoo (2D / 2.5D / 3D) | `source/model`, `scripts/diagnosis/baselines/README.md` |
| TotalSegmentator, LungMask | **missing**: empty repo clones, no weights, packages not installed |
| Python environment | `requirements.txt` not installed here (`torch`, `numpy`, `nibabel`, `pandas`) |

So the runnable path today is: install the requirements, then build a dataset profile. What
is still `blocked` needs data that has not been delivered (the Turkey cohort).
`python run.py plan <experiment>` checks the artifacts (manifests, masks, weights,
checkpoints) on the current machine and names whatever is missing.

## Quick start

```bash
# this workspace: these are already the defaults (configs/paths.yaml), exporting is optional;
# mount the raw release first: gcsfuse --implicit-dirs --only-dir Stanford_INSPECT_dataset pe-study /mnt/Stanford_INSPECT_dataset
export PE_CLOUD_ROOT=/mnt/pe-project/outputs
export PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset
export PE_DERIVED_ROOT=/mnt/pe-project/outputs/derived
export PE_LOCAL_CACHE_ROOT=/mnt/pe-project/cache

python3 -m pip install -r requirements.txt && python3 -m pip install -e .

python3 run.py preflight data.dataset.smoke_30                         # 1. check paths
python run.py run data.dataset.smoke_30 # 2. technical smoke
python run.py run data.dataset.test_500_sample --allow-full # 3. rehearsal
python3 run.py plan baseline.resnet18_3d                                # 4. what is left
PROFILE=smoke_30 bash scripts/data/eda.sh                                    # 5. describe the cohort
python tools/baselines/smoke_pipeline.py                                # 6. whole baseline pipeline on synthetic data (CPU)
```

Everything the pipeline produces goes to the git-ignored `/mnt/pe-project/outputs` on the VM
disk — cohorts under `derived/datasets/<profile>/`, caches under `derived/cache/<profile>/`,
runs under `pe-project/outputs/`, the same layout as `gs://pe-study/pe-storage`. The raw
release at `/mnt/Stanford_INSPECT_dataset` (a gcsfuse mount) is only ever read. The full
500-sample and full-cohort recipes are in [Test → full](#test--full).

`gcloud storage rsync --recursive /mnt/pe-project/outputs gs://pe-study/pe-storage` backs the outputs up
(swap the two paths to restore them on a fresh disk); the pipeline itself never reads the
bucket, and until a backup new work exists only on the VM disk.

## The CLI

```bash
python run.py list                        # every experiment, grouped by pipeline stage
python run.py show baseline.resnet18_3d   # what it is, what it needs, what it produces
python run.py plan baseline.resnet18_3d   # dependency chain, marked READY / MISSING / BLOCKED
python run.py preflight baseline.resnet18_3d --gpus 0
python run.py dry  baseline.resnet18_3d --gpus 0,1
python run.py run  baseline.resnet18_3d --gpus 0,1   # train one run; the grids use tools/baselines/
```

`run.py` is the normal interface and works the same on Windows, WSL and Linux. It holds no
scientific logic: it resolves a semantic name from
[`configs/experiments.yaml`](configs/experiments.yaml) and delegates to `tools/preflight.py`,
`tools/launch.py` or the entry's own runner (zero-shot). Those tools remain available
directly when you want them.

Experiments are named for what they are — `data.dataset.full_inspect`, `data.roi`,
`diag.zeroshot.penet`, `baseline.ctfm_frozen_3d`. The internal `experiment.id` in each config is
the output-directory key and is readable too — `SEG_pseudo_anatomy`, `DX_zeroshot_penet`,
`DX_base_resnet18_3d` — so a path under `outputs/` says what produced it. Pre-refactor ids are
recorded as `legacy_id` in the registry for traceability.

`run.py plan` never executes prerequisites. It tells you what is missing; you decide what to
run.

[`scripts/`](scripts/README.md) holds thin wrappers grouped by stage. They contain no
scientific logic — they just turn environment variables and flags into `run.py` /
`tools/` calls:

```bash
python run.py run baseline.resnet18_3d --set data.profile=test_500_sample --gpus 0
```

## The four independent axes

Every run is one point in a four-dimensional space, and the four dimensions are deliberately
independent — otherwise no two results are comparable.

| axis | set with | values |
|---|---|---|
| **dataset** | `data.profile` / `PROFILE` | `smoke_30`, `test_500_sample`, `full_inspect` |
| **method** | which experiment / model you run | one run config per arm |
| **encoder weight** | `model.backbone` × `encoder.init_source` | baseline zoo × `pretrained` (or `--scratch`) |
| **run scope** | `--patient-id` / `--max-cases` / `--allow-full` | generation stages |

None is a code path. Each is a config value, and the run id is stamped with whatever deviates
from the config's baseline, so combinations never overwrite each other:

```text
DX_base_resnet18_3d                                   baseline
DX_base_resnet18_3d__ds_test_500_sample               other dataset profile
DX_base_resnet18_3d__ds_test_500_sample__bb_resnet50_3d   profile and backbone changed
```

The baseline grids (`tools/baselines/run_case.py`) add their own layer on top:
`<model>__<head>__frac<PPP>[__v<variant>]/official_seed<S>/`.

## Pipeline

```text
0 data_preprocessing -> 1 segmentation -> ROI masks      (data artifacts: QC, analysis)
                     -> 2 silver_label
                     -> CT-FM feature cache  ------+
                     -> 3 zero-shot (PENet, RADAR) |
                     -> 4 baselines  <-------------+   diagnosis + image-only prognosis
```

Everything derived lives under `/mnt/pe-project/outputs` in this workspace: cohorts under
`derived/datasets/<profile>/`, every run under `pe-project/outputs/<stage>/<experiment.id>/`.
The raw release stays read-only at `/mnt/Stanford_INSPECT_dataset`.

| stage | wrapper | reads | writes |
|---|---|---|---|
| **0 data preprocessing** | `scripts/data/preprocessing.sh` | the read-only INSPECT release | `derived/datasets/<profile>/`: task manifests, `data_quality.md`, `dataset.json`, `logs.txt`; CT/EHR caches in `derived/cache/<profile>/` |
| **1 segmentation** | `scripts/data/segmentation.sh`, then `scripts/data/roi.sh` | `manifests/ctpa.csv` | `outputs/segmentation/SEG_pseudo_anatomy/` (26 masks/study incl. lung lobes and sides + QC manifest), then `outputs/roi/ROI_anatomy_and_controls/` (ROI1–ROI8 + matched random controls) |
| **2 silver label** | `scripts/data/silver_labels.sh` | `manifests/reports.csv` | `outputs/silver_label/<method>/silver_labels.csv` + `silver_label_confidence.csv` + `logs/run.log` |
| **3 zero-shot** | `scripts/diagnosis/zero_shot/{penet,radar}.sh` | task manifests, released weights | `outputs/diagnosis/DX_zeroshot_*` (no weight is updated) |
| **4 baselines** | `scripts/diagnosis/baselines/exp0*_*/`, `scripts/prognosis/baselines/exp0*.sh --label <outcome>`; `prepare_ctfm_cache.sh` first for `ctfm_frozen_3d` | task manifests (or `manifests/ct_fm/*.csv`), public backbone weights | `outputs/<diagnosis\|prognosis>/BASE/<profile>/<task>/` (runs + per-experiment summaries) |

`run.py plan <experiment>` prints this chain for one arm and marks each link
READY / MISSING / BLOCKED. It never runs a prerequisite for you.

- **DATASET** — three profiles, one implementation. `data.dataset.smoke_30` samples 30
  candidates (10 per official split) before eligibility/CT checks, so the final cohort may be
  smaller; `data.dataset.test_500_sample` is the 500-patient rehearsal;
  `data.dataset.full_inspect` is the whole eligible cohort. They inherit the identical
  eligibility, integrity, adjudication, manifest and preprocessing contract from
  `source/data/profiles/_common.yaml`. Each build writes one `data_quality.md`
  with task label/missing counts, `dataset.json`, `logs.txt`, and patient-linked
  exclusions in `manifests/exclusions.csv` next to the task manifests.
- **DATA** — support artifacts built on top of a profile: pseudo-anatomy masks
  (`data.segmentation`, TotalSegmentator with a LungMask lung-Dice cross-check), ROI masks and
  volume-matched random controls (`data.roi`), and report-derived silver labels
  (`data.silver.medgemma`). They are QC / analysis artifacts; no trainable model reads them.
  There is no segmentation fine-tuning: without expert masks, the masks stay pseudo-labels and
  are labelled as such.
- **ZERO-SHOT** — released PENet and RADAR applied without training
  (`diag.zeroshot.{penet,radar}`, own runners `tools/tasks/zeroshot_{penet,radar}.py`).
- **BASELINES** — 20 image encoders (2D / 2.5D slice-MIL, 3D, CT-FM LoRA / frozen) with an
  MLP or KAN head, PE +/- and image-only prognosis, in four experiments: exp01 (all 20
  models), exp02 (training-set size 25/50/75/100%), exp03 (MLP vs KAN), exp04 (slice-MIL
  ablation). Official split only, repeated with seeds; no k-fold.

Per-stage detail lives in [`docs/`](docs/); the experiment plan and paper tables are in
[docs/04_experiments.md](docs/04_experiments.md). `python run.py list` is the authoritative
inventory and marks every unresolved contract.

## Architecture

Every trainable run is one model, `task.architecture: baseline_classifier`
(`source/model/classifier.py`, built by `source/engine/factory.py:build_task_model`, which
rejects any other architecture). Only the encoder and the head change between arms, so a
difference between two results is a difference in representation, data or head, never in the
surrounding code.

```text
              CTPA volume [B,1,128,128,128]   (or cached CT-FM features)
                          |
            (1) ENCODER          source/model/2D_model, source/model/3D_model,
                          |      source/components/encoders/image (CT-FM)
                 global embedding [B,F]
                          |
            (2) optional standardizer   head.standardize_inputs (z-score fit on train)
                          |
            (3) SHARED PROJECTION       Dropout -> Linear(F, 64) -> LayerNorm
                          |
            (4) HEAD                    MLP or KAN   (source/model/head)
                          |
                PE +/-  or  one prognosis outcome
```

- **Encoders.** 2D / 2.5D slice-MIL (`resnet18`, `convnext`, `vit`, `swin`; 32 axial
  slices, gated-attention / mean / max pooling, `model.mil.slice_selection` uniform or
  center) and 3D (`resnet18/50`, `densenet121`, `convnext`, `vit`, `swin`, `nnmamba`,
  `mamba_mae`, `vmamba`, `penet`, `ctfm_lora`, `ctfm_frozen`), registered in
  `source/model/registry.py` and `configs/components/backbones.yaml`.
- **PEFT.** `full`, `frozen` or `lora` (`source/components/peft/`), applied to the encoder
  only; the projection and head are always trained.
- **Prognosis** is image-only (`task.modalities: [image]`), one binary outcome per run.

Detail and tensor shapes: [docs/02_models.md](docs/02_models.md).

### sPESI

Stage 0 computes the simplified PESI from the same MEDS/OMOP events the EHR block reads and
writes it next to the manifests (it is a dataset artifact; no trainable arm consumes it).
Components, thresholds and vital codes follow the INSPECT release's own scorer
(`third_party/repos/INSPECT_public/ehr/4_compute_pesi_score.py`). `source/clinical/spesi.py`
is verified against that function directly.

```text
derived/cache/<profile>/clinical/spesi_features.csv        spesi, spesi_high_risk, spesi_computable, per study
derived/cache/<profile>/clinical/spesi_components.csv      the six resolved components with code and event-time provenance
derived/cache/<profile>/clinical/spesi_status.json         scored cases by split, score distribution, missing components
derived/cache/<profile>/clinical/spesi_mapping_audit.json  the effective code set each component resolved to
```

Six criteria, one point each: age > 80, active cancer, chronic cardiopulmonary disease,
pulse >= 110, systolic BP < 100, SpO2 < 90. **A study missing any component is left
unscored** rather than partially summed, so an all-missing case can never look low-risk.

Two contract points live in `configs/clinical/spesi_mapping.yaml`:

- INSPECT expands its two comorbidity SNOMED concepts through FEMR's ontology. This project
  has no ontology object, so each comorbidity also lists ICD-10-CM prefixes, expanded against
  the release's own `metadata/codes.parquet` at build time. The resulting code list is written
  into the audit, so the effective code set is reviewable rather than hidden.
- INSPECT reads events up to **two days after** the scan. The default here is strict
  pre-index (`event_window_after_index_hours: 0`), consistent with the rest of stage 0. Set it
  to `48` to reproduce INSPECT's number exactly, and expect a higher completion rate.

## The baseline experiments

```bash
bash scripts/diagnosis/baselines/prepare_weights.sh                   # fetch weights once
bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh                # CT-FM cache, once per profile (ctfm_frozen_3d)
GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp01_baselines/run_all.sh
GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh
bash scripts/prognosis/baselines/exp03_head_ablation.sh --label 12_month_PH --gpus 0,1
bash scripts/tool/run_baseline_grid.sh exp04_slice_ablation all --variants "center mean max" --dry-run
```

One protocol for every arm (`configs/components/baselines.yaml`): effective batch 4, at most
100 epochs, early stopping after 15 epochs without a better validation AUROC, `best.ckpt` =
best validation AUROC, `compute.precision: auto`, seeded DataLoader; Youden threshold on
validation; 2000-sample patient bootstrap CI on test; Grad-CAM of 3 correct + 3 wrong test
cases in `visualize/`. Each experiment folder gets `summary.md`, `summary_pretty.csv`,
`summary_ensemble.csv` (seed-averaged predictions merged on `study_id`), `summary.csv` and
`summary_raw.csv`. Full guide: [scripts/diagnosis/baselines/README.md](scripts/diagnosis/baselines/README.md).

### The frozen CT-FM arm

`ctfm_frozen_3d` keeps the public CT-FM encoder frozen and trains only the projection and
head on pooled CT-FM features. The features are computed once per study by
`bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh` (wraps
`tools/data/build_ctfm_cache.py`; `PROFILE`, `GPUS`, `WORKERS`, `VERIFY_CACHE=1`,
`REBUILD_CACHE=1`) into `derived/cache/<profile>/ct_fm/` and `manifests/ct_fm/*.csv`. Older
`DX_ctfm_frozen` / `PR_ctfm_frozen_*` output folders came from a removed code path and are
historical only.

Representations are chosen on **validation** performance, never on the test split.

## Layout

```text
pe-project/
├── run.py                  the human entrypoint (list / show / plan / preflight / dry / run)
├── scripts/                thin wrappers around run.py and tools/ (see scripts/README.md)
│   ├── data/               preprocessing / segmentation / roi / silver_labels / eda
│   ├── diagnosis/          zero_shot/ (PENet, RADAR), baselines/ (exp01-exp04, prepare_ctfm_cache.sh)
│   ├── prognosis/          baselines/: exp01-exp04 for one outcome (--label)
│   └── tool/               shared code only: use_gcs_storage, _flags, run_baseline_grid
├── configs/
│   ├── experiments.yaml    the experiment registry: names, questions, requirements, status
│   ├── components/         preset catalogs: backbones, encoders, baselines, tasks, silver
│   ├── runs/               one file per runnable experiment, grouped by pipeline stage
│   │   ├── 00_data/{dataset,segmentation,roi,silver}/  01_foundation/zero_shot/
│   │   └── 02_diagnosis/baselines/{2D,2_5D,3D}/
│   ├── compute/            GPU default or CPU
│   └── paths.yaml          data/output roots
├── source/
│   ├── data/               everything about the cohort, in one package:
│   │   ├── profiles/       the three active dataset profiles, as data not code
│   │   ├── build/          stage 0: raw INSPECT -> eligible cohort -> manifests, EHR, sPESI, caches
│   │   └── *.py            runtime: paths, PyTorch Dataset, manifest reading, preflight, training fractions
│   ├── clinical/           sPESI scoring (used by the dataset build)
│   ├── model/              BaselineClassifier, 2D / 3D encoders, MLP / KAN heads
│   ├── components/         image-encoder contract + CT-FM, feature standardizer, PEFT
│   └── ...                 training engine, metrics, QC, silver, ROI, segmentation
├── analysis/               EDA over a built dataset profile (see analysis/README.md)
├── tools/                  Python CLIs per domain (see tools/README.md)
├── docs/                   7 guides: overview (README), data, models, training, experiments + paper tables, running, code map
└── third_party/            upstream clones and local weights (see third_party/README.md)
```

A run config selects named catalog presets such as `components/baselines.yaml#diagnosis`; it
never inherits from another run. See [`configs/README.md`](configs/README.md) for the
compact layout and selector syntax.

## Install

Python 3.11 recommended. From the project root:

```bash
conda create -n pe311 python=3.11 -y
conda activate pe311
python -m pip install --upgrade pip setuptools wheel
```

Install the PyTorch build matching your CUDA/driver first, then:

```bash
pip install -r requirements.txt
pip install -e .
```

The support pipeline also needs the pinned upstream CLIs:

```bash
pip install -e third_party/repos/TotalSegmentator
pip install -e third_party/repos/lungmask
```

`third_party/repos/*` are git submodules pinned to upstream commits, so a fresh clone has them
empty: run `git submodule update --init` (or clone with `--recurse-submodules`) before the
install lines above will do anything. The dataset build needs none of them — it uses only
`numpy`, `nibabel` and the standard library.

Third-party sources and weights are managed per [third_party/README.md](third_party/README.md).

## Data and environment

```powershell
$env:PE_CLOUD_ROOT = "E:\PE_NU"
$env:PE_LOCAL_CACHE_ROOT = "E:\PE_NU\cache"
cd E:\PE_NU\source\pe-project
```

```bash
export PE_CLOUD_ROOT=/mnt                                  # project at $PE_CLOUD_ROOT/pe-project
export PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset   # only if the release is elsewhere
```

`PE_CLOUD_ROOT` is the one variable everything hangs off. `PE_RAW_INSPECT_ROOT` and
`PE_DERIVED_ROOT` are optional overrides for when the read-only release or the derived tree
does not live under it; an exported override wins over the template in `configs/paths.yaml`.

With the default config this resolves to:

```text
${PE_CLOUD_ROOT}/data/Stanford_INSPECT_dataset   raw, READ ONLY
${PE_CLOUD_ROOT}/data/derived/datasets/<profile> one directory per dataset profile
${PE_CLOUD_ROOT}/pe-project/outputs              all experiment outputs
```

In this workspace `configs/paths.yaml` sets `PE_CLOUD_ROOT` to `/mnt/pe-project/outputs`, so
derived data resolves to `/mnt/pe-project/outputs/derived` and outputs to
`/mnt/pe-project/outputs/pe-project/outputs` without exporting anything. The only mount is the
raw release, which you mount yourself with gcsfuse at `/mnt/Stanford_INSPECT_dataset`.

### Building a dataset

Nothing downstream runs until one of the dataset profiles exists. The build reads the raw release,
never writes to it, and requires an explicit scope like every other generation stage:

```bash
# smoke: thirty patients, ten from each official split
python run.py run data.dataset.smoke_30

# the 500-patient rehearsal cohort
python run.py run data.dataset.test_500_sample --allow-full

# the full cohort
python run.py run data.dataset.full_inspect --allow-full
```

Each profile writes, under `${PE_DERIVED_ROOT}/datasets/<profile>/`:

```text
data_quality.md     one-page report: cohort funnel (samples left after every step),
                    task label/missing counts, QC and clinical readiness
logs.txt            terminal output from the preprocessing wrapper
dataset.json        full provenance, including the preprocessing fingerprint
manifests/          ctpa.csv, diagnosis.csv, prognosis*.csv, paired_reports.csv, reports.csv
manifests/exclusions.csv                 patient/study IDs excluded with reasons
manifests/ct_fm/*.csv                    CT-FM copies of the task manifests (prepare_ctfm_cache.sh)
```

`derived/cache/<profile>/volumes/` is the shared RAS cache;
`derived/cache/<profile>/ct_fm/` holds per-study CT-FM features (SPL, 3x1x1 mm, 24x128x128 patches).
`derived/cache/<profile>/clinical/` retains EHR and sPESI features/provenance.

Manifests carry at least `patient_id, study_id, split, image_path`; `split` is one of
`train | validation | test | external`, and the build fails if a patient appears in two.
EHR readiness columns and the sPESI score column are merged into `ctpa.csv`, `diagnosis.csv`
and the prognosis manifests.

There is **no** `*_mask_path` column: anatomy masks are a stage-1 artifact and live in the ROI
run (`roi/ROI_anatomy_and_controls/roi_manifest.csv`).

**The split is preserved, never created.** INSPECT's official `train/valid/test` assignment is
carried through unchanged (`valid` → `validation`); no tool creates a new split. The baseline
zoo's training-fraction manifests (exp02, `tools/baselines/run_case.py` →
`source/data/experiment_splits.py`) only subsample the official train patients and never
touch validation or test; there is no k-fold. Never select checkpoints or thresholds on test.

## Test → full

Every stage runs against either dataset profile. Nothing is pilot-only: the profile is a
config value, not a code path, so the 500-patient rehearsal exercises exactly the code the
full run will take.

### The 500-sample test, end to end

```bash
export PE_CLOUD_ROOT=/mnt/pe-project/outputs
export PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset
export PE_DERIVED_ROOT=/mnt/pe-project/outputs/derived
export PE_LOCAL_CACHE_ROOT=/mnt/pe-project/cache
export PROFILE=test_500_sample

python run.py run data.dataset.test_500_sample --allow-full # cohort + manifests + clinical
python run.py run data.segmentation --allow-full --gpus 0 # 19 pseudo-anatomy masks/study
python run.py run data.roi --allow-full # ROI1..ROI8 + random controls
python run.py run data.silver.medgemma --allow-full --gpus 0 # accepted silver labels

bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh # CT-FM cache for ctfm_frozen_3d
GPUS=0 bash scripts/diagnosis/baselines/exp01_baselines/run_all.sh # diagnosis baselines
bash scripts/prognosis/baselines/exp01_baselines.sh --label 1_month_mortality --gpus 0 # prognosis
```

Read `derived/datasets/test_500_sample/data_quality.md` before anything else: its **cohort
funnel** table is the per-step sample count (release → governance → eligibility → CT
integrity → sampling → scope → preprocessing → final), followed by the label distribution per
split and the exclusion counts by rule.

### The full run

Identical commands with `PROFILE=full_inspect` (the default), and
`python run.py run data.dataset.full_inspect --allow-full` for stage 0. Check a
training arm first without starting it:

```bash
python run.py preflight baseline.resnet18_3d --set data.profile=full_inspect --gpus 0
bash scripts/tool/run_baseline_grid.sh exp01_baselines all --dry-run
```

`--patient-id` / `--max-cases` / `--max-reports` / `--allow-full` apply to the **generation**
stages — dataset build, segmentation, ROI, silver labels, zero-shot — where `run.py` requires
an explicit scope so a full run is never accidental:

```bash
python run.py run data.segmentation  --gpus 0 --patient-id PATIENT_001
python run.py run data.roi                    --max-cases 5
python run.py run data.silver.medgemma --gpus 0 --max-reports 10
python run.py run data.silver.medgemma --gpus 0,1 --allow-full
```

A **training** stage has no per-case limit, and none was invented: its scope *is* the dataset
profile. Rehearse a training arm on `PROFILE=smoke_30 EPOCHS=1`, or check a grid without
starting it with `ACTION=preflight` / `--dry-run`.

`--allow-full` must always be explicit. Smoke evaluations write to their own scope and never
overwrite a full evaluation.

## GPUs

- Training uses PyTorch DDP, one process per GPU (`--gpus 0,1,2,3`).
- The baseline grids spread cases over a GPU pool (`GPUS`, `--runs-per-gpu` /
  `JOBS_PER_GPU`, `GPUS_PER_JOB`; `scripts/tool/run_baseline_grid.sh`).
- Silver generation shards reports across ranks; the medgemma method loads MedGemma on every rank,
  so check VRAM before a full run.
- Segmentation shards studies across `--gpus` without DDP; ROI construction is CPU-parallel
  via `roi.workers`.

## Rules that keep results trustworthy

- No stage runs its own prerequisites. `run.py plan` reports; you decide.
- Research outputs never live in the source tree — only under the project output root.
- One `experiment.id` per scientific configuration. `--resume` only where the stage supports
  it; `--overwrite` only when you mean to discard a run.
- Review SEG/ROI/silver QC before using them.
- Silver labels are never evaluation labels.
- One model, many encoders. Never fork the model per checkpoint kind.
- Checkpoint selection and thresholds come from validation, never from test.
- `tools/baselines/summarize.py` aggregates finished baseline runs into tables.

## Nothing is faked

Unresolved contracts — the Turkey cohort, expert segmentation annotations — are listed in
`python run.py plan` and make preflight fail on purpose. Fill them from the real artifacts;
do not guess.

Two places where the honest answer is narrower than the label suggests:

- **The frozen CT-FM arm fits a head.** The public checkpoint carries no PE head, so no metric
  exists without one. `ctfm_frozen_3d` freezes the backbone and fits only the projection and
  head on cached CT-FM features. No pretrained weight is updated — but this is a trained head
  on frozen features, not zero-shot classification, and must not be reported as one (the
  zero-shot arms are `diag.zeroshot.penet` / `diag.zeroshot.radar`).
- **Training stages have no per-case limit.** The trainer has no batch cap and none was added,
  because a `--max-cases` for training would be a pilot-only code path. A training arm's scope
  is its dataset profile.

This checkout has no `tests/` suite. Preflight, `dry`, `plan`, `tools/baselines/smoke_pipeline.py`
(synthetic data) and one-patient generation runs are the end-to-end checks.

## Where things are

There is exactly one place for each thing: run configs in `configs/runs/`, shared fragments in
`configs/components/`, the experiment index in `configs/experiments.yaml`, library code in
`source/`, CLIs in `tools/`, cohort definitions in `source/data/profiles/`, cohort code in
`source/data/build/`. No parallel copy of the pipeline. The shell scripts under
`scripts/` are wrappers around `run.py` and `tools/`, not a second definition of an experiment.

Pre-refactor experiment ids are recorded as `legacy_id` in `configs/experiments.yaml`, which
is enough to trace an old note or result folder to the arm that replaced it.

CLI detail: [tools/README.md](tools/README.md). Config conventions:
[configs/README.md](configs/README.md).
