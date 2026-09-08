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
source scripts/use_gcs_storage.sh  # this workspace: bucket-backed derived data and outputs

python3 -m pip install -r requirements.txt && python3 -m pip install -e .

python3 run.py preflight data.dataset.smoke_30                         # 1. check paths
bash scripts/0_data_preprocessing/build_smoke_30.sh                    # 2. technical smoke
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_test_500_sample.sh # 3. rehearsal
python3 run.py plan diag.anatomy.concat                                 # 4. what is left
```

Everything the pipeline produces goes to `/mnt/pe-storage` — cohorts under
`/mnt/pe-storage/derived/datasets/<profile>/`, runs under
`/mnt/pe-storage/pe-project/outputs/`. Nothing is written into the source tree, and the raw
release at `/mnt/Stanford_INSPECT_dataset` is only ever read. The full 500-sample and
full-cohort recipes are in [Test → full](#test--full).

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
0 data_preprocessing -> 1 segmentation -> 2 silver_label -> 3 shared_encoder
                                    \                              |
                                     \-------- ROI masks ----------+--> 4 diagnosis
                                                                    +--> 5 prognosis
                                                                          |
                                                                          v
                                                                    6 counterfactual
```

Everything derived lives under `/mnt/pe-storage` in this workspace: cohorts under
`/mnt/pe-storage/derived/datasets/<profile>/`, every run under
`/mnt/pe-storage/pe-project/outputs/<stage>/<experiment.id>/`. The code stays local at
`/mnt/pe-project` and the raw release stays read-only at `/mnt/Stanford_INSPECT_dataset`.

| stage | wrapper | reads | writes |
|---|---|---|---|
| **0 data preprocessing** | `scripts/0_data_preprocessing/build_*.sh` | the read-only INSPECT release | `derived/datasets/<profile>/`: `manifests/*.csv`, physical CT cache `volumes/*.npy` plus geometry/patch sidecars, `clinical/*`, `data_quality.{md,json}`, `audit/*`, `dataset.json` |
| **1 segmentation** | `scripts/1_segmentation/totalsegmentator.sh`, then `roi.sh` | `manifests/ctpa.csv` | `outputs/segmentation/SEG_pseudo_anatomy/` (19 masks/study + QC manifest), then `outputs/roi/ROI_anatomy_and_controls/` (ROI1–ROI8 + matched random controls) |
| **2 silver label** | `scripts/2_silver_label/sl0*.sh` | `manifests/reports.csv` | `outputs/silver_label/SL_*/labels.parquet` + `audit.jsonl` |
| **3 shared encoder** | `scripts/3_shared_encoder/<n>_<stage>/*.sh` | `manifests/ctpa.csv`, `manifests/paired_reports.csv`, silver labels | `outputs/pretraining/{dapt,alignment,silver}/<id>/best.ckpt` with full lineage |
| **4 diagnosis** | `scripts/4_diagnosis/*.sh` | `manifests/diagnosis.csv`, ROI masks, silver labels, an encoder checkpoint | `outputs/diagnosis/DX_*/` (`best.ckpt`, `result.json`, predictions) |
| **5 prognosis** | `scripts/5_prognosis/*.sh` | `manifests/prognosis.csv`, `clinical/ehr_features.csv`, `clinical/pesi_features.csv`, ROI masks, an encoder checkpoint | `outputs/prognosis/PR_*/` |
| **6 counterfactual** | `scripts/6_counterfactual/*.sh` | the frozen `DX_anatomy_concat` checkpoint + ROI masks | `outputs/counterfactual/CF_*/` (paired deltas, no retraining) |

`run.py plan <experiment>` prints this chain for one arm and marks each link
READY / MISSING / BLOCKED. It never runs a prerequisite for you.

- **DATASET** — three profiles, one implementation. `data.dataset.smoke_30` is the small
  all-split technical check; `data.dataset.test_500_sample` is the 500-patient rehearsal;
  `data.dataset.full_inspect` is the whole eligible cohort. They inherit the identical
  eligibility, integrity, adjudication, manifest and preprocessing contract from
  `source/dataset/profiles/_common.yaml`. Each build writes `data_quality.md/json`,
  `dataset.json`, and detailed findings under `audit/` next to the manifests.
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

## Architecture

Diagnosis and prognosis are the **same four pieces**, in the same order, so a difference
between two results is a difference in data or weights, never in code. Each piece is one
module and one config key; swapping one never touches the others.

```text
                    CTPA volume  [B,1,128,128,128]
                            |
              (1) SHARED ENCODER            source/components/encoders/image/*
                            |               model.backbone x encoder.init_source
                    feature map [B,C,D,H,W]  -- ONE forward pass over the whole volume
                            |
       +-------------+------+------+-------------+
    global         heart      pa           lung          mask-pooled views of that map
       |             |         |             |           source/components/roi/pooling.py
              (2) ORGAN ADAPTERS            source/components/adapters/organ.py
       |             |         |             |           one adapter per branch -> expert_dim
       |             |         |             |
       |             |         |             |     (prognosis only, added at the end)
       |             |         |             |            ehr        pesi
       |             |         |             |             |          |
       +-------------+----+----+-------------+-------------+----------+
                          |
              (3) PREDICTION HEAD + (4) FUSION       source/components/fusion/*
                          |
                    PE +/-  or  30-day mortality
```

**(1) Shared encoder.** The whole CTPA is encoded **once**. Heart / PA / lung are *pooled
views of that one feature map*, not separate crops through separate encoders — so the
branches cost nothing extra and cannot disagree about what they saw. `model.backbone`
(`ct_fm` / `ct_clip` / `totalfm`) and `encoder.init_source` (`pretrained` / `dapt` / `c0` /
`silver`) are independent config values; the model object is identical for all of them.

**(2) Organ adapters.** `OrganAdapterBank` gives every branch — including `global` — its own
small adapter (`bottleneck_mlp`, `residual` or `lora`) mapping the pooled feature to one
shared width (`task.expert_dim`, default 128). A branch whose mask is missing for a patient
is zeroed and flagged unavailable, and stays flagged all the way into fusion. Configure with
`configs/components/adapter/organ.yaml`; `organ_adapter.include_global: false` drops the
whole-volume branch, `task.regions: []` drops the organ branches and leaves a global-only
model.

**(3) Prediction head.** A linear head per target (`DiagnosisHeads`) or one small MLP
(`PrognosisHead`). Where the head sits depends on the fusion type, and only on that.

**(4) Fusion.** One config key, `fusion.type`, three interchangeable behaviours:

| `fusion.type` | what is combined | how |
|---|---|---|
| `concat_mlp` | features | mask the unavailable branches, concatenate, one MLP, then one head bank |
| `soft_moe` | features | a learned per-patient router weights the branches; the router is masked and renormalized for unavailable branches, then one head bank |
| `late_logit` | **decisions** | each branch gets its **own** head bank; the branch logits are averaged with masked, renormalized weights (`learned: true` trains a softmax over branches) |

Under `late_logit` and `soft_moe` a missing branch is *renormalized away*, not fed as zeros
— that is the whole point of building the fusion comparison as three cells of one grid.
Exactly one of the two prediction paths is constructed, so a `late_logit` model has no unused
feature-fusion MLP and a `concat_mlp` model has no unused per-branch heads.

### Diagnosis — imaging only

`source/tasks/diagnosis/model.py`. Primary target `pe_present`. Diagnosis **never** sees EHR
or PESI, by design: its number has to be attributable to the scan. Optional auxiliary heads
attach accepted-silver findings to the branch whose anatomy they describe (RV findings and
septal bowing → heart, clot location and acuity → PA, effusion / fibrosis / emphysema →
lung), which is what `configs/components/task/diagnosis_organ_silver.yaml` encodes. Silver
labels are supervision only; they are never evaluation labels.

```bash
DATASET=test_500_sample bash scripts/4_diagnosis/global_single.sh              # global only
DATASET=test_500_sample bash scripts/4_diagnosis/anatomy_full.sh              # + organ branches, concat
DATASET=test_500_sample bash scripts/4_diagnosis/anatomy_silver_late_logit.sh # late-logit fusion
DATASET=test_500_sample bash scripts/4_diagnosis/anatomy_silver_soft_moe.sh   # MoE fusion
```

### Prognosis — imaging plus clinical, fused at the end

`source/tasks/prognosis/model.py`. Primary target `mortality_30d` on the confirmed-acute-PE
cohort. The imaging half is byte-for-byte the diagnosis stack; `ehr` and `pesi` enter as two
more branches **at the fusion stage only**, each through its own encoder and its own adapter
to the same width. `task.modalities` selects which branches exist, which is exactly the
seven-arm modality ablation:

```bash
DATASET=test_500_sample bash scripts/5_prognosis/pesi_only.sh          # pesi
DATASET=test_500_sample bash scripts/5_prognosis/clinical_only.sh      # ehr
DATASET=test_500_sample bash scripts/5_prognosis/clinical_pesi.sh      # ehr + pesi
DATASET=test_500_sample bash scripts/5_prognosis/image_only.sh         # image
DATASET=test_500_sample bash scripts/5_prognosis/image_pesi.sh         # image + pesi
DATASET=test_500_sample bash scripts/5_prognosis/image_clinical.sh     # image + ehr
DATASET=test_500_sample bash scripts/5_prognosis/image_clinical_pesi.sh   # all three
DATASET=test_500_sample bash scripts/5_prognosis/anatomy_soft_moe.sh      # + organ branches, MoE
```

A tabular-only arm builds no image encoder at all, so it needs no backbone and no masks.

### PESI / sPESI

PESI and sPESI are a **clinical gate, not a join**. INSPECT ships raw MEDS/OMOP events, not a
validated score, so stage 0 always writes an audit and only *sometimes* writes a score:

```text
clinical/pesi_mapping_audit.json   per-component code / source / unit / timing review
clinical/pesi_status.json          blocked | built | built_no_complete_cases | disabled
clinical/pesi_features.csv         ONLY when approved: pesi, spesi, pesi_class, computability flags
```

`source/clinical/pesi.py` implements the eleven-component original PESI, its five risk
classes and sPESI, and returns `None` — never a guess — when any component is missing.
`configs/clinical/pesi_spesi_mapping.yaml` is the contract; today it is `pending`, so a build
ends normally with `pesi_status.json: blocked` and no score is fabricated. Every prognosis arm
with `pesi` in `task.modalities` fails preflight until a clinical steward approves all eleven
components **and** supplies an approved study-level `pesi.components_table`. See
[docs/BLOCKERS.md](docs/BLOCKERS.md) §6.

### Where the anatomy masks come from

There is no `*_mask_path` column in any stage-0 manifest, and none is invented. Anatomy masks
are a stage-1 artifact, read straight from the stored ROI run through `data.roi_manifest` +
`data.roi_mask_ids` (`configs/components/data/anatomy_masks.yaml`), with the same
PASS/SUSPICIOUS QC filter the students and counterfactuals use:

```text
heart -> ROI2 (strict heart)    pa -> ROI4 (PA tree)    lung -> ROI6 (lung parenchyma)
```

So a branch, the region erased from it and the student trained on it alone all refer to
exactly the same voxels. Anatomy-aware arms therefore need `data.segmentation` **and then**
`data.roi`.

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
│   ├── 3_shared_encoder/{0_pretrained_eval,1_dapt,2_alignment,3_silver_encoder}/
│   └── 4_diagnosis/  5_prognosis/  6_counterfactual/
├── configs/
│   ├── experiments.yaml    the experiment registry: names, questions, requirements, status
│   ├── components/         reusable fragments: backbone registry, encoder initialization,
│   │                       dapt, task, fusion, adapter, data (anatomy mask source),
│   │                       alignment, silver, training
│   ├── clinical/           PESI/sPESI mapping contract, EHR feature policy
│   ├── runs/               one file per runnable experiment, grouped by pipeline stage
│   │   ├── 00_data/{dataset,silver}/  01_foundation/  02_representation/
│   │   ├── 03_diagnosis/  04_prognosis/  05_anatomy_analysis/
│   │   └── 90_deferred/
│   ├── compute/            GPU default or CPU
│   └── paths.yaml          data/output roots
├── source/
│   ├── data_preprocessing/ raw INSPECT -> eligible cohort -> manifests (one implementation)
│   ├── dataset/            the three active dataset profiles, as data not code
│   ├── clinical/           PESI/sPESI scoring, clinical preprocessing, the EHR encoder
│   ├── components/         encoders, organ adapters, ROI pooling, fusion, PEFT
│   ├── tasks/              diagnosis / prognosis / contour models and heads
│   └── ...                 data, training engine, metrics, QC, silver, ROI, segmentation
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

In this workspace, first mount `gs://pe-study/pe-storage` yourself at
`/mnt/pe-storage`, then use `source scripts/use_gcs_storage.sh`. The helper never mounts
anything; it keeps code local while resolving derived data to `/mnt/pe-storage/derived`
and outputs to `/mnt/pe-storage/pe-project/outputs`.

### Building a dataset

Nothing downstream runs until one of the dataset profiles exists. The build reads the raw release,
never writes to it, and requires an explicit scope like every other generation stage:

```bash
# smoke: thirty patients, ten from each official split
bash scripts/0_data_preprocessing/build_smoke_30.sh

# the 500-patient rehearsal cohort
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_test_500_sample.sh

# the full cohort
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_full_inspect.sh
```

Each profile writes, under `${PE_DERIVED_ROOT}/datasets/<profile>/`:

```text
data_quality.md     one-page report: cohort funnel (samples left after every step),
                    label distribution per split, QC and data loss, clinical readiness
data_quality.json   machine-readable form of that report
dataset.json        full provenance, including the preprocessing fingerprint
manifests/          ctpa.csv, diagnosis.csv, prognosis.csv, paired_reports.csv, reports.csv
volumes/            the preprocessed volume cache; each NPY has a physical-geometry
                    sidecar and, when configured, a world-coordinate patch manifest
clinical/           leakage-safe EHR artifacts, PESI mapping audit/status, and approved scores
audit/              detailed exclusions, integrity, split and cache-QC findings
```

Manifests carry at least `patient_id, study_id, split, image_path`; `split` is one of
`train | validation | test | external`, and the build fails if a patient appears in two.
EHR readiness columns and, once approved, the PESI/sPESI score columns are merged into
`ctpa.csv`, `diagnosis.csv` and `prognosis.csv`, so a training arm reads clinical values from
the manifest and never from a second table.

There is **no** `*_mask_path` column: anatomy masks are a stage-1 artifact and are read from
the ROI run instead — see [Where the anatomy masks come from](#where-the-anatomy-masks-come-from).

**The split is preserved, never created.** INSPECT's official `train/valid/test` assignment is
carried through unchanged (`valid` → `validation`). `tools/data/create_split.py` exists only
for a cohort that arrives with no split of its own, and is never invoked implicitly. Never
select checkpoints or thresholds on test.

## Test → full

Every stage runs against either dataset profile. Nothing is pilot-only: `DATASET` is a config
value, not a code path, so the 500-patient rehearsal exercises exactly the code the full run
will take.

```bash
DATASET=test_500_sample MAX_CASES=10 ...    # smoke, generation stages
DATASET=test_500_sample ALLOW_ALL=1  ...    # the whole 500-patient subset
DATASET=full_inspect    ALLOW_ALL=1  ...    # the whole cohort
```

### The 500-sample test, end to end

```bash
source scripts/use_gcs_storage.sh
export DATASET=test_500_sample

ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_test_500_sample.sh   # cohort + manifests + clinical
ALLOW_ALL=1 GPUS=0 bash scripts/1_segmentation/totalsegmentator.sh       # 19 pseudo-anatomy masks/study
ALLOW_ALL=1        bash scripts/1_segmentation/roi.sh                    # ROI1..ROI8 + random controls
ALLOW_ALL=1 GPUS=0 bash scripts/2_silver_label/sl02_hybrid.sh            # accepted silver labels

GPUS=0 bash scripts/3_shared_encoder/1_dapt/dino.sh                      # adapt the shared encoder
GPUS=0 bash scripts/3_shared_encoder/2_alignment/image_report.sh         # -> C0

GPUS=0 bash scripts/4_diagnosis/anatomy_full.sh                          # diagnosis
GPUS=0 bash scripts/5_prognosis/image_clinical_pesi.sh                   # prognosis
GPUS=0 bash scripts/6_counterfactual/remove_pa.sh                        # necessity analysis
```

Read `derived/datasets/test_500_sample/data_quality.md` before anything else: its **cohort
funnel** table is the per-step sample count (release → governance → eligibility → CT
integrity → sampling → scope → preprocessing → final), followed by the label distribution per
split and the exclusion counts by rule.

### The full run

Identical commands with `DATASET=full_inspect` and
`ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_full_inspect.sh` for stage 0. Check a
training arm first without starting it:

```bash
DATASET=full_inspect ACTION=preflight GPUS=0 bash scripts/4_diagnosis/anatomy_full.sh
DATASET=full_inspect ACTION=dry SMOKE=1 GPUS=0 bash scripts/4_diagnosis/anatomy_full.sh
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
