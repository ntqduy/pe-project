# PE Project — 3D CTPA

Research framework for representation learning and pulmonary embolism (PE) tasks on 3D CTPA.
Every stage has its own config, checkpoint lineage, QC and outputs. Only real models and real
pipelines: nothing is mocked, and unresolved data contracts fail loudly instead of being
guessed.

## Where this repository is right now

`python run.py list` shows 59 experiments. Four are `ready`; the rest are `blocked` on an
artifact that does not exist yet, and say so rather than guessing.

| | state |
|---|---|
| Raw Stanford INSPECT release | present, read-only. 23,248 studies / 19,405 patients |
| Dataset profiles | code and configs complete, **not built yet** — build one first |
| Falcon, MedGemma (silver labels) | weights staged locally, ids and revisions recorded |
| CT-FM, CT-CLIP, TotalFM | **missing**: empty repo clones, no weights, adapter contract empty |
| TotalSegmentator, LungMask | **missing**: empty repo clones, no weights, packages not installed |
| Python environment | `requirements.txt` not installed here (`torch`, `numpy`, `nibabel`, `pandas`) |

So the runnable path today is: install the requirements, then build a dataset profile. Every
image experiment stays blocked until a backbone contract is filled in from a real inspected
checkpoint — see [docs/BLOCKERS.md](docs/BLOCKERS.md).

## Quick start

```bash
export PE_CLOUD_ROOT=/mnt
export PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset   # only if the release lives elsewhere

pip install -r requirements.txt && pip install -e .

python run.py preflight data.dataset.test_500_sample                              # 1. check
DATASET=test_500_sample MAX_CASES=10 bash scripts/0_data_preprocessing/build_test_500_sample.sh   # 2. smoke
DATASET=test_500_sample ALLOW_ALL=1  bash scripts/0_data_preprocessing/build_test_500_sample.sh   # 3. cohort
python run.py plan diag.anatomy.concat                                            # 4. what is left
```

## The CLI

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

[`scripts/`](scripts/README.md) holds one thin wrapper per experiment, grouped by stage. They
contain no scientific logic — they just turn environment variables into `run.py --set`
overrides:

```bash
DATASET=test_500_sample ENCODER_SOURCE=dapt GPUS=0 bash scripts/4_diagnosis/anatomy_full.sh
```

## The four independent axes

Every run is one point in a four-dimensional space, and the four dimensions are deliberately
independent — otherwise no two results are comparable.

| axis | set with | values |
|---|---|---|
| **dataset** | `data.profile` / `DATASET` | `test_500_sample`, `full_inspect` |
| **method** | which experiment you run | one config each |
| **encoder weight** | `model.backbone` × `encoder.init_source` | `ct_fm`/`ct_clip`/`totalfm` × `pretrained`/`dapt`/`c0`/`silver` |
| **run scope** | `--patient-id` / `--max-cases` / `--allow-full` | generation stages |

None is a code path. Each is a config value, and the run id is stamped with whatever deviates
from the config's baseline, so combinations never overwrite each other:

```text
DX_anatomy_concat                                              baseline
DX_anatomy_concat__enc_dapt                                    DAPT initialization
DX_anatomy_concat__ds_test_500_sample__bb_ct_clip__enc_silver  all three changed
```

## Pipeline

```text
DATASET -> DATA (masks, ROIs, silver) -> SHARED ENCODER -> DIAGNOSIS / PROGNOSIS -> ANATOMY ANALYSIS
```

- **DATASET** — two profiles, one implementation. `data.dataset.full_inspect` is the whole
  eligible cohort after filtering and QC; `data.dataset.test_500_sample` is that same cohort
  reduced to 500 patients, sampled **patient-level inside the official INSPECT split**. Both
  inherit the identical eligibility, integrity, adjudication, manifest and preprocessing
  contract from `source/dataset/profiles/_common.yaml`, so there is no pilot-only and no
  full-only scientific code. Each build writes `exclusions.csv`, `integrity.json`,
  `split_audit.json` and `dataset.json` next to the manifests.
- **DATA** — support artifacts built on top of a profile: pseudo-anatomy masks
  (`data.segmentation`, TotalSegmentator with a LungMask lung-Dice cross-check), ROI crops and
  volume-matched random controls (`data.roi`), and report-derived silver labels
  (`data.silver.*`: SL00 MedGemma, SL01 rules+Falcon, SL02 adjudicated hybrid). Support
  artifacts are inputs, never results, and only `accepted` silver rows are ever trained on.
  There is no `SEG02`, no segmentation fine-tuning: without expert masks, the masks stay
  pseudo-labels and are labelled as such.
- **SHARED ENCODER** — the public backbone (CT-FM / CT-CLIP / TotalFM) is evaluated as a
  baseline *before* adaptation, then adapted: DAPT (none / MAE / DINO / SimCLR / anatomy) →
  image-report alignment → **C0** → silver adaptation → **C_silver**. Backbone and DAPT method
  are independent, and DAPT loads the original public weights itself — no foundation stage has
  to be run first. Side branches: external RSPECT transfer and C_silver; neither is a
  prerequisite for anything downstream. CT-CLIP zero-shot is a documented, deliberately
  unavailable contract, not an implemented arm.
- **DIAGNOSIS** — image only, PE +/-; single-task vs multitask, native vs accepted-silver
  organ supervision, global vs anatomy-aware, concat vs late-logit vs Soft-MoE. Diagnosis
  never uses EHR or PESI. The same diagnosis model is also the instrument for the
  representation comparison below.
- **PROGNOSIS** — confirmed-acute-PE cohort, 30-day mortality; seven-arm modality ablation,
  then global vs anatomy-aware image and the three fusion types.
- **ANATOMY ANALYSIS** — necessity (`anatomy.remove_*`: one frozen model, region erased, paired
  deltas, no retraining) and sufficiency (`anatomy.*_student[_kd]`: ROI-only students from C0,
  with and without frozen-teacher distillation), both against a volume-matched random control.

Contour and the concept bottleneck are deferred; see `configs/runs/90_deferred/`.

Full detail: **[docs/PIPELINE.md](docs/PIPELINE.md)**. Row-per-experiment table:
**[docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md)**. Everything unresolved:
**[docs/BLOCKERS.md](docs/BLOCKERS.md)**.

## Which weights are best for diagnosis?

This is the question the shared-encoder stage exists to answer, and it is answered with **one**
model, not four. Same architecture, same dataset, same split, same hyperparameters — only the
encoder initialization changes:

```bash
DATASET=test_500_sample ENCODER_SOURCE=pretrained bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=dapt       bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=c0         bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=silver     bash scripts/4_diagnosis/anatomy_full.sh
```

There is deliberately no `diagnosis_pretrained_model.py` / `diagnosis_dapt_model.py` /
`diagnosis_silver_model.py`: four model files would be four architectures, and any difference
between the results could then be blamed on the code rather than on the representation.
`encoder.init_source` is a config variable; the model is the same object. Prognosis arms with
an image encoder take the same variable.

Every checkpoint and every result records `lineage.encoder_init_source`, the source checkpoint
and its SHA-256, the backbone, the DAPT method, the alignment flag, the silver source and the
dataset profile it was adapted on — so a number always says where it came from.

### The fixed probes

`probe.diag` and `probe.prog` put a fixed head on a frozen encoder, as a cheap comparison
instrument for the same question. The split, preprocessing, head, optimizer, epochs, seed and
metrics are fixed by
[`configs/components/training/probe.yaml`](configs/components/training/probe.yaml) so that
scores are comparable across checkpoints:

```bash
python run.py run probe.diag --gpus 0 --set encoder.init_source=dapt
```

Representations are chosen on **validation** performance, never on pretraining loss alone and
never on the test split.

## Layout

```text
pe-project/
├── run.py                  the human entrypoint (list / show / plan / preflight / dry / run)
├── scripts/                thin wrappers around run.py, one per experiment
│   ├── _lib.sh             env (DATASET / BACKBONE / ENCODER_SOURCE / SCOPE) -> run.py
│   ├── 0_data_preprocessing/  1_segmentation/  2_silver_label/
│   ├── 3_shared_encoder/{pretrained_eval,dapt,alignment,silver_encoder}/
│   └── 4_diagnosis/  5_prognosis/  6_counterfactual/
├── configs/
│   ├── experiments.yaml    the experiment registry: names, questions, requirements, status
│   ├── components/         reusable fragments: backbone registry, encoder initialization,
│   │                       dapt, task, fusion, adapter, alignment, silver, training
│   ├── runs/               one file per runnable experiment, grouped by pipeline stage
│   │   ├── 00_data/{dataset,silver}/  01_foundation/  02_representation/
│   │   ├── 03_diagnosis/  04_prognosis/  05_anatomy_analysis/
│   │   └── 90_deferred/
│   ├── compute/            GPU default or CPU
│   └── paths.yaml          data/output roots
├── source/
│   ├── data_preprocessing/ raw INSPECT -> eligible cohort -> manifests (one implementation)
│   ├── dataset/            the two active dataset profiles, as data not code
│   └── ...                 models, data, training engine, metrics, QC, pipeline logic
├── tools/                  Python CLIs per domain (see tools/README.md)
├── docs/                   PIPELINE, EXPERIMENT_MAP, BLOCKERS
└── third_party/            upstream clones and local weights (see third_party/README.md)
```

A run config inherits only from `components/` — never from another run config. Two levels
maximum, so you can read one file and know what it does. A script never contains an
experiment: if a change belongs in a wrapper, it belongs in a config instead.

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

**Those two directories are currently empty**, as are `CT-FM`, `CT-CLIP` and `TotalFM`. Clone
them at the commits pinned in [`third_party/versions.yaml`](third_party/versions.yaml) before
the install lines above will do anything. The dataset build needs none of them — it uses only
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

### Building a dataset

Nothing downstream runs until one of the two profiles exists. The build reads the raw release,
never writes to it, and requires an explicit scope like every other generation stage:

```bash
# smoke: ten patients end to end
DATASET=test_500_sample MAX_CASES=10 bash scripts/0_data_preprocessing/build_test_500_sample.sh

# the 500-patient rehearsal cohort
DATASET=test_500_sample ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_test_500_sample.sh

# the full cohort
DATASET=full_inspect ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_full_inspect.sh
```

Each profile writes, under `data/derived/datasets/<profile>/`:

```text
manifests/ctpa.csv  diagnosis.csv  prognosis.csv  paired_reports.csv  reports.csv
exclusions.csv      every study dropped, and the rule that dropped it
integrity.json      corrupted / missing CT findings
split_audit.json    patient-leakage and official-split preservation
dataset.json        full provenance, including the preprocessing fingerprint
volumes/            the preprocessed volume cache
```

Manifests carry at least `patient_id, study_id, split, image_path`; `split` is one of
`train | validation | test | external`, and the build fails if a patient appears in two.

**The split is preserved, never created.** INSPECT's official `train/valid/test` assignment is
carried through unchanged (`valid` → `validation`). `tools/data/create_split.py` exists only
for a cohort that arrives with no split of its own, and is never invoked implicitly. Never
select checkpoints or thresholds on test.

## Test → full

Every stage runs against either dataset profile. Nothing is pilot-only.

```bash
DATASET=test_500_sample MAX_CASES=10 ...    # smoke, generation stages
DATASET=test_500_sample ALLOW_ALL=1  ...    # the whole 500-patient subset
DATASET=full_inspect    ALLOW_ALL=1  ...    # the whole cohort
```

`--patient-id` / `--max-cases` / `--max-reports` / `--allow-full` apply to the **generation**
stages — dataset build, segmentation, ROI, silver labels, counterfactual inference — where
`run.py` requires an explicit scope so a full run is never accidental:

```bash
python run.py run data.segmentation  --gpus 0 --patient-id PATIENT_001
python run.py run data.roi                    --max-cases 5
python run.py run data.silver.hybrid --gpus 0 --max-reports 10
python run.py run anatomy.remove_pa  --gpus 0 --patient-id PATIENT_001   # smoke, isolated output
python run.py run data.silver.hybrid --gpus 0,1 --allow-full
```

A **training** stage has no per-case limit, and none was invented: its scope *is* the dataset
profile. Rehearse a training arm on `DATASET=test_500_sample`, and check a run without
starting it with `ACTION=preflight` or `ACTION=dry` (`SMOKE=1` additionally pins
`training.epochs=1`).

`--allow-full` must always be explicit. Smoke evaluations write to their own scope and never
overwrite a full evaluation.

## GPUs

- Training uses PyTorch DDP, one process per GPU (`--gpus 0,1,2,3`).
- Silver generation shards reports across ranks; SL02 loads both LLMs on every rank, so check
  VRAM before a full run.
- Segmentation shards studies across `--gpus` without DDP; ROI construction is CPU-parallel
  via `roi.workers`.
- Several independent experiments at once: `tools/launch_parallel.py` splits
  `parallel.devices` into disjoint groups of `parallel.gpus_per_job` and runs the listed jobs
  in waves. There is no batch file in the repository — write one, then `--dry-run` it first:

  ```yaml
  # sweep.yaml
  parallel:
    enabled: true          # explicit opt-in; the launcher refuses to run without it
    devices: [0, 1, 2, 3]
    gpus_per_job: 1
    jobs:
      - config: configs/runs/03_diagnosis/anatomy/single_concat.yaml
      - config: configs/runs/03_diagnosis/baseline/global_single.yaml
  ```

  ```bash
  python tools/launch_parallel.py --config sweep.yaml --dry-run
  ```

  Each job must name a distinct config: per-job `args` accept only selection flags and
  `--resume`, not `--set`. An encoder-initialization comparison therefore runs sequentially
  through `scripts/`, not through this launcher.

## Rules that keep results trustworthy

- No stage runs its own prerequisites. `run.py plan` reports; you decide.
- Research outputs never live in the source tree — only under the project output root.
- One `experiment.id` per scientific configuration. `--resume` only where the stage supports
  it; `--overwrite` only when you mean to discard a run.
- Review SEG/ROI/silver QC before training anything that consumes them.
- Only `accepted` silver rows are trained on. Silver labels are never evaluation labels.
- Keep the pretrained backbone, each DAPT checkpoint, C0 and C_silver as separate
  checkpoints with their own lineage. A diagnosis result must be able to name the one it
  started from.
- One diagnosis model, many initializations. Never fork the model per checkpoint kind.
- Checkpoint selection and thresholds come from validation, never from test.
- `python tools/build_summary.py` aggregates existing `result.json` files.

## Nothing is faked

Unresolved contracts — public encoder factory/feature_dim, report embedding columns, the 32
EHR variables, the external RSPECT manifest, expert segmentation annotations — are listed in
[docs/BLOCKERS.md](docs/BLOCKERS.md) and make preflight fail on purpose. Fill them from the
real artifacts; do not guess.

Two places where the honest answer is narrower than the label suggests:

- **`pretrained_eval` fits a head.** The public checkpoints carry no PE head, so no metric
  exists without one. These arms freeze the backbone (`peft.method: frozen`) and fit only the
  fixed probe head from `components/training/probe.yaml`. No pretrained weight is updated, and
  the probe contract is identical for every checkpoint — but this is a linear probe, not
  zero-shot classification, and must not be reported as one.
- **Training stages have no per-case limit.** The trainer has no batch cap and none was added,
  because a `--max-cases` for training would be a pilot-only code path. A training arm's scope
  is its dataset profile.

There is no test suite in this repository yet, and no pytest workflow. Preflight, `dry`,
`plan` and one-patient generation runs are the available checks.

## Where things are

There is exactly one place for each thing: run configs in `configs/runs/`, shared fragments in
`configs/components/`, the experiment index in `configs/experiments.yaml`, library code in
`source/`, CLIs in `tools/`, cohort definitions in `source/dataset/profiles/`, cohort code in
`source/data_preprocessing/`. No parallel copy of the pipeline. The shell scripts under
`scripts/` are wrappers around `run.py`, not a second definition of an experiment.

The pre-refactor experiment ids are recorded as `legacy_id` in the registry and in
[docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md), which is enough to trace an old note or
result folder to the arm that replaced it.

[`info.md`](info.md) is a Vietnamese companion overview. Its `source/` module descriptions
still broadly hold, but its layout and command sections describe the pre-refactor repository
and predate `scripts/`, `source/dataset/`, `source/data_preprocessing/` and the
backbone/encoder config variables. Trust this README and `docs/` over it.

CLI detail: [tools/README.md](tools/README.md). Config conventions:
[configs/README.md](configs/README.md).
