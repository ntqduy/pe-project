# `scripts/`

The general launcher is still `run.py`, but the CT-FM frozen baseline also has explicit
task wrappers. The wrappers only select a reviewed config/cohort; preprocessing and model
logic remain in `source/` and `tools/`.

```text
scripts/
├── run_preprocessing.sh    stage 0 end to end for one dataset profile
├── run_segmentation.sh     TotalSegmentator + LungMask masks and QC
├── run_roi.sh               ROI1-ROI8 and matched random controls
├── run_silver_labels.sh     report-derived accepted/abstained silver labels
├── run_silver_adaptation.sh optional silver-supervised encoder adaptation
├── run_ctfm_diagnosis.sh   CT-FM frozen + trainable MLP for PE diagnosis
├── run_ctfm_prognosis_all.sh  CT-FM frozen + MLP prognosis on all eligible patients
├── run_ctfm_prognosis_pe.sh   CT-FM frozen + MLP prognosis on PE-positive patients
├── run_ctfm_frozen.sh      shared prepare/train/evaluate implementation
├── run_ctfm_anatomy.sh     CT-FM frozen + organ adapters + concat/Soft-MoE proposal
├── run_zeroshot_penet.sh   released PENet applied unchanged, no training
└── ana.sh                  EDA over a built dataset profile (see analysis/README.md)
```

## Building a dataset profile

```bash
export PE_CLOUD_ROOT=/mnt/pe-storage
export PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset

PROFILE=smoke_30 bash scripts/run_preprocessing.sh        # rehearse first
ACTION=preflight bash scripts/run_preprocessing.sh        # check the full build
bash scripts/run_preprocessing.sh                         # full_inspect, hours
```

`PROFILE` picks `smoke_30`, `test_500_sample` or `full_inspect`; `ACTION` picks
`run`, `dry`, `preflight`, `plan` or `show`; `OVERWRITE=1` rebuilds over an existing
profile and `MAX_CASES=N` limits the scope instead of building everything.

The preprocessing pipeline applies eligibility/noise filters, missing/corrupt-volume
checks, patient label adjudication, official split/leakage checks, and writes audit
artifacts. The CT-FM cache adds a second per-volume QC pass and records failures in
`ct_fm_frozen/preprocessing_failures.csv` and `ct_fm_frozen/dropped_rows.csv`; it never
moves a row between train/validation/test.

## Running an experiment

```bash
python run.py list                      # every experiment, grouped by pipeline stage
python run.py show  diag.anatomy.concat # what it is, what it needs, what it produces
python run.py plan  diag.anatomy.concat # dependency chain: READY / MISSING / BLOCKED
python run.py preflight diag.anatomy.concat --gpus 0
python run.py dry   diag.anatomy.concat --gpus 0
python run.py run   diag.anatomy.concat --gpus 0
```

## What the old environment variables map to

| old wrapper variable | now |
|---|---|
| `DATASET=<profile>` | `--set data.profile=<profile>` |
| `BACKBONE=<name>` | `--set model.backbone=<name>` |
| `ENCODER_SOURCE=<stage>` | `--set encoder.init_source=<stage>` |
| `ENCODER_CHECKPOINT=<path>` | `--set encoder.checkpoint=<path>` |
| `ENCODER_EXPERIMENT=<id>` | `--set encoder.source_experiment=<id>` |
| `SMOKE=1` | `--set training.epochs=1` |
| `GPUS=0,1` | `--gpus 0,1` |
| `ACTION=<verb>` | the `run.py` subcommand (`show`, `plan`, `preflight`, `dry`, `run`) |
| `ALLOW_ALL=1` | `--allow-full` |
| `MAX_CASES=N` / `MAX_REPORTS=N` / `PATIENT_ID=id` | `--max-cases N` / `--max-reports N` / `--patient-id id` |
| `RESUME=1` / `OVERWRITE=1` | `--resume` / `--overwrite` |
| `SET="a=1 b=2"` | `--set a=1 --set b=2` |

For the explicit CT-FM wrappers:

```bash
# Technical rehearsal: existing small profile, one epoch, one GPU.
PROFILE=smoke_30 EPOCHS=1 GPUS=0 bash scripts/run_ctfm_diagnosis.sh

# Full diagnosis: 50-epoch budget, early stop after 10 validation stalls, two GPUs.
PROFILE=full_inspect EPOCHS=50 EARLY_STOPPING=10 GPUS=0,1 \
  bash scripts/run_ctfm_diagnosis.sh

# Proposal: same frozen public CT-FM, then train only organ adapters/fusion/head.
TASK=diagnosis FUSION=concat PROFILE=full_inspect GPUS=0,1 \
  bash scripts/run_ctfm_anatomy.sh
TASK=diagnosis FUSION=moe PROFILE=full_inspect GPUS=0,1 \
  bash scripts/run_ctfm_anatomy.sh
TASK=prognosis COHORT=pe FUSION=concat PROFILE=full_inspect GPUS=0,1 \
  bash scripts/run_ctfm_anatomy.sh
TASK=prognosis COHORT=pe FUSION=moe PROFILE=full_inspect GPUS=0,1 \
  bash scripts/run_ctfm_anatomy.sh

# Prognosis variants.
PROFILE=full_inspect GPUS=0,1 bash scripts/run_ctfm_prognosis_all.sh
PROFILE=full_inspect TARGET=12_month_PH GPUS=0,1 \
  bash scripts/run_ctfm_prognosis_pe.sh
```

`ACTION` for these wrappers is `prepare`, `train`, `evaluate`, `all`, `preflight`, or
`dry`. `all` prepares the CT-FM cache, trains the MLP, then evaluates the untouched test
split. The task output is intentionally compact: `resolved_config.yaml`, `result.json`, and
one `epoch_<epochs_run>/` bundle containing `checkpoint/{best.ckpt,last.ckpt}`, `logs.txt`,
`result.csv`, `training_curves.pdf`, and validation-only `preview/` heatmaps for up to five
patients. `result.csv` has one wide row per split (`train`, `validation`, `test`) and target;
the test row additionally contains patient-bootstrap CI columns. For prognosis this includes
all seven endpoints and records the cohort (`all_comers` or `pe_positive_only`) in every row.
`logs.txt` keeps the experiment header, split audit, epoch summaries and completion/error
messages; launcher command lines and low-level launcher diagnostics are not stored there.
Diagnosis reports AUROC/AUPRC, sensitivity, specificity, F1, Brier and
the validation-selected threshold; prognosis reports the same plus calibration metrics and
a calibration curve. Prognosis runs default to all seven endpoints; set `TARGET` to run
one endpoint independently, with a separate output ID. Early stopping selects the best
checkpoint using negative validation loss; the final clinical metrics are computed only
afterward by `evaluate.py` on the test split. CT preprocessing QC is kept separately in
the CT-FM cache's `dataset.json`, `preprocessing_failures.csv` and `dropped_rows.csv`; it
is not mixed into task clinical metrics.

Generation stages (dataset, segmentation, ROI, silver labels, counterfactual) still need an
explicit scope: pass `--max-cases N`, `--patient-id <id>` or `--allow-full`. A training stage
has no per-case limit — its scope *is* the dataset profile, so rehearse on
`--set data.profile=test_500_sample` and use `preflight` / `dry` to check a run without
starting it.

## Storage

The pipeline writes only under `PE_CLOUD_ROOT`. Mount `gs://pe-study/pe-storage` at
`/mnt/pe-storage`, then export it yourself — nothing in this repository mounts anything:

```bash
export PE_CLOUD_ROOT=/mnt/pe-storage
export PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset
export PE_DERIVED_ROOT=/mnt/pe-storage/derived
export PE_LOCAL_CACHE_ROOT=/mnt/pe-project/cache
```

## Zero-shot PENet

The released PENet is a trained CTPA PE classifier, so applying it unchanged is an
external baseline rather than another arm to train.

```bash
pip install opencv-python-headless        # cv2.INTER_AREA slice resize, required

CHECK_SLICE_ORDER=1 bash scripts/run_zeroshot_penet.sh   # run this first
SLICE_ORDER=<winner> bash scripts/run_zeroshot_penet.sh  # then the full test split
```

Run the check first. A 3-D convolution is sensitive to slice order inside a window and
the PENet repository never records which direction its pre-sorted volumes used, so the
check scores 50 cases both ways; the wrong direction lands near chance. Getting this
wrong produces a plausible-looking but meaningless baseline with no error message.

`PROFILE`, `MAX_CASES`, `RESTRICT_TO`, `AGGREGATE` (`max`/`mean`), `GPUS`, `CHECKPOINT`
and `OVERWRITE` are the other knobs. Pass `RESTRICT_TO` whenever the row has to sit in
the same table as an arm that is not computable for every case.

## Sweeps

Matrix and ablation sweeps were loops over config overrides, not separate code paths.
Write the loop inline:

```bash
for V in global_only global_heart global_pa global_lung full_moe full_no_router; do
  python run.py run ablation.arch.$V --set data.profile=full_inspect --gpus 0
done
```

For a cohort × endpoint sweep, drive the config directly and give each cell its own
output directory:

```bash
for CH in PE_positive all_patient; do
  for T in 1_month_mortality 12_month_mortality 12_month_PH; do
    python tools/tasks/train_task.py \
      --config configs/runs/04_prognosis/modality/image_ehr.yaml \
      --set data.cohort=$CH --set data.ehr_profile=EHR_0_h --set task.primary_target=$T \
      --set experiment.output_id=$CH/EHR_0_h/$T/image_clinical
  done
done
```

See [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) for the full result-table recipes.
