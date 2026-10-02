# PE Project — 3D CTPA

Research framework for representation learning and pulmonary embolism (PE) tasks on 3D CTPA.
Every stage has its own config, checkpoint lineage, QC and outputs. Only real models and real
pipelines: nothing is mocked, and unresolved data contracts fail loudly instead of being
guessed.

## Where this repository is right now

`python run.py list` is the authoritative experiment inventory; blocked entries state the
missing artifact or unresolved contract instead of guessing.

| | state |
|---|---|
| Raw Stanford INSPECT release | present, read-only. 23,248 studies / 19,405 patients |
| Dataset profiles | code and configs complete, **not built yet** — build one first |
| MedGemma (silver labels) | weights staged locally, id and revision recorded |
| CT-FM | staged: strict load of the HF feature extractor (see `third_party/versions.yaml`) |
| Baseline zoo (2D / 2.5D / 3D) | `source/model`, `scripts/diagnosis/baselines/README.md` |
| TotalSegmentator, LungMask | **missing**: empty repo clones, no weights, packages not installed |
| Python environment | `requirements.txt` not installed here (`torch`, `numpy`, `nibabel`, `pandas`) |

So the runnable path today is: install the requirements, then build a dataset profile. The
CT-FM contract is filled in, so image arms are `ready`; what is still `blocked` needs a real
contract that has not been delivered (EHR column selection, the Turkey cohort).
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
python3 run.py plan diag.anatomy.concat                                 # 4. what is left
PROFILE=smoke_30 bash scripts/data/eda.sh                                    # 5. describe the cohort
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

Experiments are named for what they are — `foundation.ct_fm_frozen.diagnosis`,
`diag.anatomy.silver.moe`, `prog.image_ehr`, `anatomy.remove_pa`,
`ablation.arch.full_moe`. The internal `experiment.id` in each config is the output-directory
key and is now readable too — `SEG_pseudo_anatomy`, `DX_anatomy_concat`, `CF_remove_pa` — so a
path under `outputs/` says what produced it. The pre-refactor ids (`SEG01`, `DX18`, `CF02`, …)
are recorded as `legacy_id` in the registry for traceability.

`run.py plan` never executes prerequisites. It tells you what is missing; you decide what to
run.

[`scripts/`](scripts/README.md) holds one thin wrapper per experiment, grouped by stage. They
contain no scientific logic — they just turn environment variables into `run.py --set`
overrides:

```bash
python run.py run diag.anatomy.concat --set data.profile=test_500_sample --set model.backbone=ct_fm --gpus 0
```

## The four independent axes

Every run is one point in a four-dimensional space, and the four dimensions are deliberately
independent — otherwise no two results are comparable.

| axis | set with | values |
|---|---|---|
| **dataset** | `data.profile` / `DATASET` | `test_500_sample`, `full_inspect` |
| **method** | which experiment you run | one config each |
| **encoder weight** | `model.backbone` × `encoder.init_source` | `ct_fm` / baseline zoo × `pretrained`/`diagnosis`/`custom` |
| **run scope** | `--patient-id` / `--max-cases` / `--allow-full` | generation stages |

None is a code path. Each is a config value, and the run id is stamped with whatever deviates
from the config's baseline, so combinations never overwrite each other:

```text
DX_anatomy_concat                                              baseline
DX_anatomy_concat__enc_diagnosis                                  diagnosis-encoder initialization
DX_anatomy_concat__ds_test_500_sample__bb_ct_fm_features__enc_diagnosis  all three changed
```

## Pipeline

```text
0 data_preprocessing -> 1 segmentation -> 2 silver_label     3 foundation (public backbone
                                    \                  |          baselines, side branch)
                                     \--- ROI masks ---+--> 4 diagnosis
                                                       +--> 5 prognosis
                                                             |
                                                             v
                                                       6 counterfactual
```

Everything derived lives under `/mnt/pe-project/outputs` in this workspace: cohorts under
`derived/datasets/<profile>/`, every run under `pe-project/outputs/<stage>/<experiment.id>/`.
The raw release stays read-only at `/mnt/Stanford_INSPECT_dataset`.

| stage | wrapper | reads | writes |
|---|---|---|---|
| **0 data preprocessing** | `scripts/data/preprocessing.sh` | the read-only INSPECT release | `derived/datasets/<profile>/`: task manifests, `data_quality.md`, `dataset.json`, `logs.txt`; CT/EHR caches in `derived/cache/<profile>/` |
| **1 segmentation** | `scripts/data/segmentation.sh`, then `scripts/data/roi.sh` | `manifests/ctpa.csv` | `outputs/segmentation/SEG_pseudo_anatomy/` (26 masks/study incl. lung lobes and sides + QC manifest), then `outputs/roi/ROI_anatomy_and_controls/` (ROI1–ROI8 + matched random controls) |
| **2 silver label** | `scripts/data/silver_labels.sh` | `manifests/reports.csv` | `outputs/silver_label/<method>/silver_labels.csv` + `silver_label_confidence.csv` + `logs/run.log` |
| **3 foundation** | `scripts/diagnosis/zero_shot/{penet,radar}.sh`, `scripts/diagnosis/foundation/ctfm_frozen.sh` (shared driver `scripts/tool/run_ctfm_frozen.sh`) | task manifests, public backbone weights | `outputs/diagnosis/DX_zeroshot_*` and the CT-FM frozen runs `DX_ctfm_*` / `PR_ctfm_*` (no encoder weight is updated) |
| **4 diagnosis** | `python run.py run diag.*`; baseline zoo `scripts/diagnosis/baselines/exp0*_*/` | `manifests/diagnosis.csv`, ROI masks, silver labels, public backbone weights | `outputs/diagnosis/DX_*/epoch_<E>/` (`checkpoint/best.ckpt`, `result.json`, predictions) |
| **5 prognosis** | `python run.py run prog.*`; `scripts/prognosis/foundation/ctfm_frozen_{all,pe}.sh` | `manifests/prognosis.csv` (EHR/sPESI columns included), ROI masks, an encoder checkpoint | `outputs/prognosis/PR_*/` |
| **6 counterfactual** | `python run.py run anatomy.remove_* --allow-full` | the frozen `DX_anatomy_concat` checkpoint + ROI masks | `outputs/counterfactual/CF_*/` (paired deltas, no retraining) |

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
  (`data.silver.medgemma`). Support
  artifacts are inputs, never results, and only `accepted` silver rows are ever trained on.
  There is no `SEG02`, no segmentation fine-tuning: without expert masks, the masks stay
  pseudo-labels and are labelled as such.
- **FOUNDATION** — the public backbone (CT-FM) is evaluated as a
  frozen baseline, alongside released PENet / RADAR applied zero-shot. Diagnosis and
  prognosis load the original public weights themselves (`encoder.init_source=pretrained`)
  — no foundation stage has to be run first.
- **DIAGNOSIS** — image only, PE +/-; single-task vs multitask, native vs accepted-silver
  organ supervision, global vs anatomy-aware, concat vs late-logit vs Soft-MoE. Diagnosis
  never uses EHR or sPESI. External test: `diag.external.turkey_test`.
- **PROGNOSIS** — confirmed-acute-PE cohort, 30-day mortality; seven-arm modality ablation,
  then global vs anatomy-aware image and the three fusion types.
- **ANATOMY ANALYSIS** — necessity (`anatomy.remove_*`: one frozen model, region erased, paired
  deltas, no retraining) against a volume-matched random control, plus the architecture
  ablation (`ablation.arch.*`).

Per-stage detail lives in [`docs/`](docs/); the experiment run plan for a paper is
[docs/04_experiments.md](docs/04_experiments.md). `python run.py list` is the authoritative
inventory and marks every unresolved contract.

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
       |             |         |             |            ehr       spesi
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
(`ct_fm`, or any baseline-zoo encoder) and `encoder.init_source` (`pretrained` / `diagnosis` /
`custom`) are independent config values; the model object is identical for all of them.

**(2) Organ adapters.** `OrganAdapterBank` gives every branch — including `global` — its own
small adapter (`bottleneck_mlp`, `residual` or `lora`) mapping the pooled feature to one
shared width (`task.expert_dim`, default 128). A branch whose mask is missing for a patient
is zeroed and flagged unavailable, and stays flagged all the way into fusion. Configure with
`configs/components/anatomy.yaml#organ_adapter`; `organ_adapter.include_global: false` drops the
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
by design: its number has to be attributable to the scan. Optional auxiliary heads
attach accepted-silver findings to the branch whose anatomy they describe (RV findings and
septal bowing → heart, clot location and acuity → PA, effusion / fibrosis / emphysema →
lung), which is what `configs/components/tasks.yaml#diagnosis_organ_silver` encodes. Silver
labels are supervision only; they are never evaluation labels.

```bash
python run.py run diag.global.single --set data.profile=test_500_sample # global only
python run.py run diag.anatomy.concat --set data.profile=test_500_sample # + organ branches, concat
python run.py run diag.anatomy.silver.late --set data.profile=test_500_sample # late-logit fusion
python run.py run diag.anatomy.silver.moe --set data.profile=test_500_sample # MoE fusion
```

### Prognosis — imaging plus clinical, fused at the end

`source/tasks/prognosis/model.py`. Primary target `mortality_30d` on the confirmed-acute-PE
cohort. The imaging half is byte-for-byte the diagnosis stack; `ehr` and `spesi` enter as
two more branches **at the fusion stage only**, each through its own encoder and its own
adapter to the same width. `task.modalities` selects which branches exist, which is exactly
the modality ablation:

```bash
python run.py run prog.spesi --set data.profile=test_500_sample # sPESI only
python run.py run prog.ehr --set data.profile=test_500_sample # ehr
python run.py run prog.image --set data.profile=test_500_sample # image
python run.py run prog.image_ehr --set data.profile=test_500_sample # image + ehr
python run.py run prog.anatomy.moe --set data.profile=test_500_sample # + organ branches, MoE
```

A tabular-only arm builds no image encoder at all, so it needs no backbone and no masks.

### sPESI

Stage 0 computes the simplified PESI from the same MEDS/OMOP events the EHR block reads.
Components, thresholds and vital codes follow the INSPECT release's own scorer
(`third_party/repos/INSPECT_public/ehr/4_compute_pesi_score.py`), so `prog.spesi` is the
clinical baseline that release reports against. `source/clinical/spesi.py` is verified
against that function directly.

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

#### Beating the INSPECT baseline, not a different question

The point of `prog.spesi` is to be the thing a CT-FM arm has to beat, so the comparison
has to be like-for-like in three ways that are easy to get wrong:

```bash
PROFILE=full_inspect

# 1. the baseline is a fixed score, not a trained model
python tools/tasks/score_baseline.py --config configs/runs/03_prognosis/modality/spesi.yaml \
  --score-column spesi --allow-full --restrict-to ${PE_DERIVED_ROOT}/cache/$PROFILE/clinical/spesi_evaluable.csv

# 2. every other arm is scored on the identical case list
python tools/tasks/evaluate.py --config configs/runs/03_prognosis/modality/image.yaml \
  --restrict-to ${PE_DERIVED_ROOT}/cache/$PROFILE/clinical/spesi_evaluable.csv --allow-full
```

- **Fixed score, not a learned arm.** `score_baseline.py` maps the raw score onto `[0,1]`
  with a monotone transform and scores it directly. Training an encoder on one scalar can
  reorder cases, which would quietly make the reference another model. `--transform`
  picks `platt` (default; calibrated on validation only, so Brier is interpretable),
  `minmax`, or `raw_sigmoid` (what INSPECT uses). Results land in
  `prognosis/PR_spesi_only/score_baseline/`.
- **One shared case list.** sPESI is not computable for every study, so without
  `--restrict-to` the baseline and the imaging arm would be scored on different patients
  and the difference between them would be uninterpretable. Stage 0 writes
  `derived/cache/<profile>/clinical/spesi_evaluable.csv` for this; it is the same thing INSPECT does inline with
  `--compare_vs_pesi`.
- **Reproducing their number.** Set `spesi.index_overrides` in the dataset profile to
  `{age_precision: fractional, age_days_per_year: 365, event_window_after_index_hours: 48}`
  and read the PE-positive cohort. Without the age override, roughly 1.8% of patients
  score differently, because INSPECT compares a fractional age against 80 and the default
  here uses completed years.

### Where the anatomy masks come from

There is no `*_mask_path` column in any stage-0 manifest, and none is invented. Anatomy masks
are a stage-1 artifact, read straight from the stored ROI run through `data.roi_manifest` +
`data.roi_mask_ids` (`configs/components/anatomy.yaml#anatomy_masks`), with the same
PASS/SUSPICIOUS QC filter the counterfactuals use:

```text
heart -> ROI2 (strict heart)    pa -> ROI4 (PA tree)    lung -> ROI6 (lung parenchyma)
```

So a branch and the region erased from it refer to exactly the same voxels. Anatomy-aware arms therefore need `data.segmentation` **and then**
`data.roi`.

## Which weights are best for diagnosis?

It is answered with **one** model. Same architecture, same dataset, same split, same
hyperparameters — only the backbone or the encoder initialization changes:

```bash
python run.py run diag.anatomy.concat --set data.profile=test_500_sample --set encoder.init_source=pretrained
python run.py run diag.anatomy.concat --set data.profile=test_500_sample --set model.backbone=resnet18_3d
python run.py run diag.anatomy.concat --set data.profile=test_500_sample \
  --set encoder.init_source=custom --set encoder.checkpoint=<path> --set encoder.source_experiment=<id>
```

There is deliberately no per-checkpoint model file: separate model files would be separate
architectures, and any difference between the results could then be blamed on the code rather
than on the representation. `encoder.init_source` (`pretrained` / `diagnosis` / `custom`) is a
config variable; the model is the same object. Prognosis arms with an image encoder take the
same variable.

Every checkpoint and every result records `lineage.encoder_init_source`, the source checkpoint
and its SHA-256, the backbone and the dataset profile — so a number always says where it came
from.

### The frozen CT-FM arms

`foundation.ct_fm_frozen.*` (configs `configs/runs/01_foundation/ct_fm_frozen/*.yaml`) keep the
public CT-FM encoder frozen (`peft.method: frozen`) as a cheap comparison instrument for the
same question. CT-FM features are computed once per study by `tools/data/build_ctfm_cache.py`
into `derived/cache/<profile>/ct_fm/` and `manifests/ct_fm/*.csv`; only the small feature
adapter, the fusion (organ adapters for the `anatomy_*` arms) and the heads are trained:

```bash
bash scripts/tool/run_ctfm_frozen.sh                            # prepare cache, train, evaluate
python run.py run foundation.ct_fm_frozen.diagnosis --gpus 0    # train only; the cache must exist
```

Representations are chosen on **validation** performance, never on the test split.

## Cross-validation and ablations

Three additions that make the study protocol runnable. All of them wrap the existing
trainer — none of them replaces it.

### Architecture ablation

Six variants, each retrained from the public backbone (`encoder.init_source=pretrained`),
changing only `task.regions` and the fusion component:

```bash
for V in global_only global_heart global_pa global_lung full_moe full_no_router; do
  python run.py run ablation.arch.$V --set data.profile=full_inspect --gpus 0
done

# check all six without training any
for V in global_only global_heart global_pa global_lung full_moe full_no_router; do
  python run.py preflight ablation.arch.$V --set data.profile=full_inspect
done
```

| Variant | `task.regions` | Fusion |
|---|---|---|
| `global_only` | `[]` | concat + MLP |
| `global_heart` | `[heart]` | concat + MLP |
| `global_pa` | `[pa]` | concat + MLP |
| `global_lung` | `[lung]` | concat + MLP |
| `full_moe` | `[heart, pa, lung]` | Soft-MoE router |
| `full_no_router` | `[heart, pa, lung]` | concat + fusion MLP |

`full_moe` vs `full_no_router` isolates the router; the four partial rows isolate each
expert. Output: `outputs/ablation/architecture/AB_arch_<variant>/`.

### EHR variable ablation

Two axes: variable set (all / common subset) × missingness indicators (on / off).

```bash
for V in all_with_missingness all_no_missingness common_with_missingness common_no_missingness; do
  python run.py preflight ablation.ehr.$V
done

python run.py run ablation.ehr.all_with_missingness \
  --set data.ehr_columns="[age,sex,hr,sbp]" --set data.ehr_columns_common="[age,sex]"
```

| Variant | Variables | `task.ehr_include_missingness` |
|---|---|---|
| `all_with_missingness` | all | `true` |
| `all_no_missingness` | all | `false` |
| `common_with_missingness` | common subset | `true` |
| `common_no_missingness` | common subset | `false` |

Turning indicators off halves the clinical encoder's input width
(`source/clinical/encoder.py`), so the model no longer knows which values were imputed.
`data.ehr_columns` and `data.ehr_columns_common` are still empty placeholders — pass real
lists through `EHR_COLUMNS` / `EHR_COLUMNS_COMMON` or fill them in
`components/tasks.yaml#prognosis_primary_cohort` first.
Output: `outputs/ablation/ehr/AB_ehr_<variant>/`.

## Layout

```text
pe-project/
├── run.py                  the human entrypoint (list / show / plan / preflight / dry / run)
├── scripts/                thin wrappers around run.py and tools/ (see scripts/README.md)
│   ├── data/               preprocessing / segmentation / roi / silver_labels / eda
│   ├── diagnosis/          zero_shot/ (PENet, RADAR), foundation/ (CT-FM), baselines/ (exp01-exp03)
│   ├── prognosis/          foundation/: CT-FM prognosis (all patients / PE-positive)
│   └── tool/               shared code only: use_gcs_storage, _flags, run_ctfm_frozen, run_baseline_grid
├── configs/
│   ├── experiments.yaml    the experiment registry: names, questions, requirements, status
│   ├── components/         eight preset catalogs: backbones, encoders, baselines,
│   │                       tasks, training, fusions, anatomy, and silver
│   ├── runs/               one file per runnable experiment, grouped by pipeline stage
│   │   ├── 00_data/{dataset,silver}/  01_foundation/
│   │   └── 02_diagnosis/  03_prognosis/  04_anatomy_analysis/
│   ├── compute/            GPU default or CPU
│   └── paths.yaml          data/output roots
├── source/
│   ├── data/               everything about the cohort, in one package:
│   │   ├── profiles/       the three active dataset profiles, as data not code
│   │   ├── build/          stage 0: raw INSPECT -> eligible cohort -> manifests, EHR, sPESI, caches
│   │   └── *.py            runtime: paths, PyTorch Dataset, manifest reading, preflight, folds
│   ├── clinical/           sPESI scoring, clinical preprocessing and the EHR encoder
│   ├── components/         encoders, organ adapters, ROI pooling, fusion, PEFT
│   ├── tasks/              diagnosis / prognosis models and heads
│   └── ...                 training engine, metrics, QC, silver, ROI, segmentation
├── analysis/               EDA over a built dataset profile (see analysis/README.md)
├── tools/                  Python CLIs per domain (see tools/README.md)
├── docs/                   7 guides: overview (README), data, models, training, experiments + paper tables, running, code map
└── third_party/            upstream clones and local weights (see third_party/README.md)
```

A run config selects named catalog presets such as `components/tasks.yaml#diagnosis`; it
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
manifests/          ctpa.csv, diagnosis.csv, prognosis.csv, paired_reports.csv, reports.csv
manifests/exclusions.csv                 patient/study IDs excluded with reasons
manifests/ct_fm/*.csv                    CT-FM copies of the task manifests (image_path -> CT-FM features)
```

`derived/cache/<profile>/volumes/` is the shared RAS cache;
`derived/cache/<profile>/ct_fm/features/` holds per-study CT-FM features (SPL, 3x1x1 mm, 24x128x128 patches).
`derived/cache/<profile>/clinical/` retains EHR and sPESI features/provenance.

Manifests carry at least `patient_id, study_id, split, image_path`; `split` is one of
`train | validation | test | external`, and the build fails if a patient appears in two.
EHR readiness columns and the sPESI score column are merged into
`ctpa.csv`, `diagnosis.csv` and `prognosis.csv`, so a training arm reads clinical values from
the manifest and never from a second table.

There is **no** `*_mask_path` column: anatomy masks are a stage-1 artifact and are read from
the ROI run instead — see [Where the anatomy masks come from](#where-the-anatomy-masks-come-from).

**The split is preserved, never created.** INSPECT's official `train/valid/test` assignment is
carried through unchanged (`valid` → `validation`); no tool creates a new split. The baseline
zoo's k-fold / training-fraction manifests (`tools/data/build_split_manifests.py`) only
re-divide the official train+validation patients and never touch the test split. Never select
checkpoints or thresholds on test.

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
export PE_CLOUD_ROOT=/mnt/pe-project/outputs
export PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset
export PE_DERIVED_ROOT=/mnt/pe-project/outputs/derived
export PE_LOCAL_CACHE_ROOT=/mnt/pe-project/cache
export DATASET=test_500_sample

python run.py run data.dataset.test_500_sample --allow-full # cohort + manifests + clinical
python run.py run data.segmentation --allow-full --gpus 0 # 19 pseudo-anatomy masks/study
python run.py run data.roi --allow-full # ROI1..ROI8 + random controls
python run.py run data.silver.medgemma --allow-full --gpus 0 # accepted silver labels

python run.py run diag.anatomy.concat --gpus 0 # diagnosis
python run.py run prog.image_ehr --gpus 0 # prognosis
python run.py run anatomy.remove_pa --gpus 0 # necessity analysis
```

Read `derived/datasets/test_500_sample/data_quality.md` before anything else: its **cohort
funnel** table is the per-step sample count (release → governance → eligibility → CT
integrity → sampling → scope → preprocessing → final), followed by the label distribution per
split and the exclusion counts by rule.

### The full run

Identical commands with `--set data.profile=full_inspect`, and
`python run.py run data.dataset.full_inspect --allow-full` for stage 0. Check a
training arm first without starting it:

```bash
python run.py preflight diag.anatomy.concat --set data.profile=full_inspect --gpus 0
python run.py dry diag.anatomy.concat --set data.profile=full_inspect --set training.epochs=1 --gpus 0
```

`--patient-id` / `--max-cases` / `--max-reports` / `--allow-full` apply to the **generation**
stages — dataset build, segmentation, ROI, silver labels, counterfactual inference — where
`run.py` requires an explicit scope so a full run is never accidental:

```bash
python run.py run data.segmentation  --gpus 0 --patient-id PATIENT_001
python run.py run data.roi                    --max-cases 5
python run.py run data.silver.medgemma --gpus 0 --max-reports 10
python run.py run anatomy.remove_pa  --gpus 0 --patient-id PATIENT_001   # smoke, isolated output
python run.py run data.silver.medgemma --gpus 0,1 --allow-full
```

A **training** stage has no per-case limit, and none was invented: its scope *is* the dataset
profile. Rehearse a training arm on `DATASET=test_500_sample`, and check a run without
starting it with `ACTION=preflight` or `ACTION=dry` (`SMOKE=1` additionally pins
`training.epochs=1`).

`--allow-full` must always be explicit. Smoke evaluations write to their own scope and never
overwrite a full evaluation.

## GPUs

- Training uses PyTorch DDP, one process per GPU (`--gpus 0,1,2,3`).
- Silver generation shards reports across ranks; the medgemma method loads MedGemma on every rank,
  so check VRAM before a full run.
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
      - config: configs/runs/02_diagnosis/anatomy/single_concat.yaml
      - config: configs/runs/02_diagnosis/global/global_single.yaml
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
- Keep every encoder checkpoint separate, with its own lineage. A diagnosis result must be
  able to name the one it started from.
- One diagnosis model, many initializations. Never fork the model per checkpoint kind.
- Checkpoint selection and thresholds come from validation, never from test.
- `python tools/build_summary.py` aggregates existing `result.json` files.

## Nothing is faked

Unresolved contracts — public encoder factory/feature_dim, the 32
EHR variables, expert segmentation annotations — are listed in
`python run.py plan` and make preflight fail on purpose. Fill them from the
real artifacts; do not guess.

Two places where the honest answer is narrower than the label suggests:

- **The frozen CT-FM arms fit a head.** The public checkpoint carries no PE head, so no metric
  exists without one. These arms freeze the backbone (`peft.method: frozen`) and fit only the
  feature adapter, fusion and heads on cached CT-FM features. No pretrained weight is updated —
  but this is a trained head on frozen features, not zero-shot classification, and must not be
  reported as one (the zero-shot arms are `diag.zeroshot.penet` / `diag.zeroshot.radar`).
- **Training stages have no per-case limit.** The trainer has no batch cap and none was added,
  because a `--max-cases` for training would be a pilot-only code path. A training arm's scope
  is its dataset profile.

This checkout has no `tests/` suite. Preflight, `dry`, `plan` and one-patient generation runs
are the end-to-end checks.

## Where things are

There is exactly one place for each thing: run configs in `configs/runs/`, shared fragments in
`configs/components/`, the experiment index in `configs/experiments.yaml`, library code in
`source/`, CLIs in `tools/`, cohort definitions in `source/data/profiles/`, cohort code in
`source/data/build/`. No parallel copy of the pipeline. The shell scripts under
`scripts/` are wrappers around `run.py`, not a second definition of an experiment.

The pre-refactor experiment ids are recorded as `legacy_id` in the registry and in
`legacy_id` in `configs/experiments.yaml`, which is enough to trace an old note or
result folder to the arm that replaced it.

CLI detail: [tools/README.md](tools/README.md). Config conventions:
[configs/README.md](configs/README.md).
