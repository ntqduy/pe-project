# PE Project — 3D CTPA

Research framework for representation learning and pulmonary embolism (PE) tasks on 3D CTPA.
Every stage has its own config, checkpoint lineage, QC and outputs. Only real models and real
pipelines: nothing is mocked, and unresolved data contracts fail loudly instead of being
guessed.

## Start here

```bash
python run.py list                     # every experiment, grouped by pipeline stage
python run.py show diag.anatomy.concat # what it is, what it needs, what it produces
python run.py plan diag.anatomy.concat # dependency chain, marked READY / MISSING / BLOCKED
python run.py preflight diag.anatomy.concat --gpus 0
python run.py dry  diag.anatomy.concat --gpus 0,1
python run.py run  diag.anatomy.concat --gpus 0,1
```

`run.py` is the normal interface and works the same on Windows, WSL and Linux. It holds no
scientific logic: it resolves a semantic name from
[`configs/experiments.yaml`](configs/experiments.yaml) and delegates to `tools/preflight.py`
and `tools/launch.py`. Those tools remain available directly when you want them.

Experiments are named for what they are — `repr.dapt.dino`, `probe.diag`,
`diag.anatomy.silver.moe`, `prog.image_ehr_pesi`, `anatomy.remove_pa`,
`anatomy.pa_student_kd`. The internal `experiment.id` in each config is the output-directory
key and is now readable too — `SEG_pseudo_anatomy`, `DX_anatomy_concat`, `CF_remove_pa` — so a
path under `outputs/` says what produced it. The pre-refactor ids (`SEG01`, `DX18`, `CF02`, …)
are recorded as `legacy_id` in the registry for traceability.

`run.py plan` never executes prerequisites. It tells you what is missing; you decide what to
run.

## Pipeline

```text
DATA -> FOUNDATION SCREENING -> REPRESENTATION ADAPTATION -> DIAGNOSIS / PROGNOSIS -> ANATOMY ANALYSIS
```

- **DATA** — INSPECT preprocessing, one explicit patient-level split, native labels, EHR/PESI,
  plus two support artifacts: anatomy masks (`data.segmentation` → `data.roi`) and
  report-derived silver labels (`data.silver.*`). Support artifacts are inputs, never results.
- **FOUNDATION SCREENING** — CT-FM, CT-CLIP, TotalFM. Image-only encoders are screened as
  frozen encoder → linear probe. CT-CLIP zero-shot is a documented, deliberately unavailable
  contract, not an implemented arm.
- **REPRESENTATION** — DAPT (none / MAE / DINO / SimCLR / anatomy) → image-report alignment
  → **C0**. Side branches: external RSPECT transfer and silver-adapted **C_silver**. Neither
  is a prerequisite for anything downstream.
- **DIAGNOSIS** — image only, PE +/-; single-task vs multitask, native vs accepted-silver
  organ supervision, global vs anatomy-aware, concat vs late-logit vs Soft-MoE. Diagnosis
  never uses EHR or PESI.
- **PROGNOSIS** — confirmed-acute-PE cohort, 30-day mortality; seven-arm modality ablation,
  then global vs anatomy-aware image and the three fusion types.
- **ANATOMY ANALYSIS** — necessity (`anatomy.remove_*`: one frozen model, region erased, paired
  deltas, no retraining) and sufficiency (`anatomy.*_student[_kd]`: ROI-only students from C0,
  with and without frozen-teacher distillation), both against a volume-matched random control.

Contour and the concept bottleneck are deferred; see `configs/runs/90_deferred/`.

Full detail: **[docs/PIPELINE.md](docs/PIPELINE.md)**. Row-per-experiment table:
**[docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md)**. Everything unresolved:
**[docs/BLOCKERS.md](docs/BLOCKERS.md)**.

## Choosing a representation: the fixed probes

`probe.diag` and `probe.prog` put a fixed head on a frozen encoder. The split, preprocessing,
head, optimizer, epochs, seed and metrics are fixed by
[`configs/components/training/probe.yaml`](configs/components/training/probe.yaml) so that
scores are comparable across checkpoints. Only the checkpoint under test and the experiment
id may change:

```bash
python run.py run probe.diag --gpus 0 \
  --set lineage.source_experiment=D_dapt_dino \
  --set lineage.source_checkpoint='${PE_CLOUD_ROOT}/pe-project/outputs/pretraining/dapt/D_dapt_dino/best.ckpt' \
  --set experiment.id=DX_probe_diagnosis_dapt_dino
```

Representations are chosen on **validation** performance, never on pretraining loss alone and
never on the test split.

## Layout

```text
pe-project/
├── run.py                  the human entrypoint (list / show / plan / preflight / dry / run)
├── configs/
│   ├── experiments.yaml    the experiment registry: names, questions, requirements, status
│   ├── components/         reusable fragments: backbone, dapt, task, fusion, adapter,
│   │                       alignment, silver, training contracts
│   ├── runs/               one file per runnable experiment, grouped by pipeline stage
│   │   ├── 00_data/  01_foundation/  02_representation/
│   │   ├── 03_diagnosis/  04_prognosis/  05_anatomy_analysis/
│   │   └── 90_deferred/
│   ├── compute/            GPU default or CPU
│   ├── parallel/           batch scheduler for independent jobs
│   └── paths.yaml          data/output roots
├── source/                 models, data, training engine, metrics, QC, pipeline logic
├── tools/                  Python CLIs per domain (see tools/README.md)
├── docs/                   PIPELINE, EXPERIMENT_MAP, BLOCKERS
└── third_party/            upstream clones and local weights (see third_party/README.md)
```

A run config inherits only from `components/` — never from another run config. Two levels
maximum, so you can read one file and know what it does.

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

Third-party sources and weights are managed per [third_party/README.md](third_party/README.md).

## Data and environment

```powershell
$env:PE_CLOUD_ROOT = "E:\PE_NU"
$env:PE_LOCAL_CACHE_ROOT = "E:\PE_NU\cache"
cd E:\PE_NU\source\pe-project
```

With the default config this resolves to:

```text
${PE_CLOUD_ROOT}/data/Stanford_INSPECT_dataset   raw, read only
${PE_CLOUD_ROOT}/data/derived                    manifests and normalized data
${PE_CLOUD_ROOT}/pe-project/outputs              all experiment outputs
```

Manifests live under `data/derived/manifests/` and need at least
`patient_id, study_id, split, image_path`. `split` is one of
`train | validation | test | external`, and a patient must not appear in two splits.
`reports.parquet` for silver generation needs `patient_id, study_id, report_id, report_text`.

If the data has no split yet, create exactly one explicit patient-level split:

```bash
python tools/data/create_split.py --input SOURCE.csv --output manifests/ctpa.csv --create-split --seed 42
```

Never create a split implicitly inside train/evaluate, and never select checkpoints on test.

## Running a single patient, a few cases, or everything

Data-generation and counterfactual stages require one explicit selection mode:

```bash
python run.py run data.segmentation  --gpus 0 --patient-id PATIENT_001
python run.py run data.roi                    --max-cases 5
python run.py run data.silver.hybrid --gpus 0 --max-reports 10
python run.py run anatomy.remove_pa  --gpus 0 --patient-id PATIENT_001   # smoke, isolated output
python run.py run data.silver.hybrid --gpus 0,1 --allow-full
```

`--allow-full` must always be explicit. Smoke evaluations write to their own scope and never
overwrite a full evaluation.

## GPUs

- Training uses PyTorch DDP, one process per GPU (`--gpus 0,1,2,3`).
- Silver generation shards reports across ranks; SL02 loads both LLMs on every rank, so check
  VRAM before a full run.
- Segmentation shards studies across `--gpus` without DDP; ROI construction is CPU-parallel
  via `roi.workers`.
- Several independent experiments at once: edit
  [`configs/parallel/experiments.yaml`](configs/parallel/experiments.yaml), then
  `python tools/launch_parallel.py --config configs/parallel/experiments.yaml --dry-run`.

## Rules that keep results trustworthy

- No stage runs its own prerequisites. `run.py plan` reports; you decide.
- Research outputs never live in the source tree — only under the project output root.
- One `experiment.id` per scientific configuration. `--resume` only where the stage supports
  it; `--overwrite` only when you mean to discard a run.
- Review SEG/ROI/silver QC before training anything that consumes them.
- Only `accepted` silver rows are trained on. Silver labels are never evaluation labels.
- Keep C0 and C_silver as separate checkpoints.
- Checkpoint selection and thresholds come from validation, never from test.
- `python tools/build_summary.py` aggregates existing `result.json` files.

## Nothing is faked

Unresolved contracts — public encoder factory/feature_dim, the real manifests and their
columns, report embedding columns, the 32 EHR variables, silver model weights, expert
segmentation annotations — are listed in [docs/BLOCKERS.md](docs/BLOCKERS.md) and make
preflight fail on purpose. Fill them from the real artifacts; do not guess.

There is no test suite in this repository yet, and no pytest workflow. Preflight, `dry`, and
one-patient runs are the available checks.

## Where things are

There is exactly one place for each thing: run configs in `configs/runs/`, shared fragments in
`configs/components/`, the experiment index in `configs/experiments.yaml`, library code in
`source/`, CLIs in `tools/`. No parallel copy of the pipeline, and no per-experiment shell
scripts — `run.py` replaced them.

The pre-refactor experiment ids are recorded as `legacy_id` in the registry and in
[docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md), which is enough to trace an old note or
result folder to the arm that replaced it.

[`info.md`](info.md) is a Vietnamese companion overview; its `source/` module descriptions
still hold, its layout and command sections describe the pre-refactor repository.

CLI detail: [tools/README.md](tools/README.md). Config conventions:
[configs/README.md](configs/README.md).
