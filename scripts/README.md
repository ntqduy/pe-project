# `scripts/`

The general launcher is still `run.py`, but the CT-FM frozen baseline also has explicit
task wrappers. The wrappers only select a reviewed config/cohort; preprocessing and model
logic remain in `source/` and `tools/`.

```text
scripts/
├── diagnosis/
│   ├── run_ctfm_diagnosis.sh     CT-FM frozen + trainable MLP for PE diagnosis
│   ├── run_zeroshot_penet.sh     released PENet applied unchanged, no training
│   └── run_zeroshot_radar.sh     released RADAR (abdominal-CT generalist) probed for PE, no training
├── prognosis/
│   ├── run_ctfm_prognosis_all.sh CT-FM frozen + MLP prognosis on all eligible patients
│   └── run_ctfm_prognosis_pe.sh  CT-FM frozen + MLP prognosis on PE-positive patients
└── tool/
    ├── use_gcs_storage.sh        storage roots (PE_*); sourced by every script, see Storage
    ├── run_ctfm_frozen.sh        shared CT-FM prepare/train/evaluate behind the task wrappers
    ├── run_preprocessing.sh      stage 0 end to end for one dataset profile
    ├── run_segmentation.sh       TotalSegmentator + LungMask masks, contact-sheet previews, QC
    ├── run_roi.sh                ROI1-ROI8 and matched random controls
    ├── run_silver_labels.sh      report-derived accepted/abstained silver labels
    └── eda.sh                    EDA over a built dataset profile (see analysis/README.md)
```

Every script takes its settings from environment variables in front of the command
(`PROFILE=smoke_30 GPUS=0 bash scripts/<folder>/<name>.sh`). Common ones: `PROFILE`
(`smoke_30` | `test_500_sample` | `full_inspect`, default `full_inspect`; `eda.sh` defaults
to `test_500_sample`), `ACTION`, `GPUS`, `OVERWRITE=1`, `PYTHON`.

## Building a dataset profile

```bash
PROFILE=smoke_30 bash scripts/tool/run_preprocessing.sh        # rehearse first
ACTION=preflight bash scripts/tool/run_preprocessing.sh        # check the full build
bash scripts/tool/run_preprocessing.sh                         # full_inspect, hours
```

`ACTION` picks `run`, `dry`, `preflight`, `plan` or `show`; `OVERWRITE=1` rebuilds over an
existing profile and `MAX_CASES=N` limits the scope instead of building everything.
Manifests, `data_quality.md`, `dataset.json` and `logs.txt` are written to
`<derived>/datasets/<profile>/`; the CT and clinical caches live in
`<derived>/cache/<profile>/`.

The preprocessing pipeline applies eligibility/noise filters, missing/corrupt-volume
checks, patient label adjudication, official split/leakage checks, and writes audit
artifacts. Its terminal output is saved in `logs.txt`; one `data_quality.md` reports
missing labels by task and QC counts. The CT-FM cache adds a second per-volume QC pass;
when failures occur it records affected IDs in `derived/cache/<profile>/ct_fm/preprocessing_failures.csv`
and `derived/cache/<profile>/ct_fm/dropped_rows.csv`. It never
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
| `REBUILD_CACHE=1` (CT-FM wrappers) | `build_ctfm_cache.py --overwrite`; `OVERWRITE=1` no longer rebuilds the feature cache |
| `SET="a=1 b=2"` | `--set a=1 --set b=2` |

For the explicit CT-FM wrappers:

```bash
# Technical rehearsal: existing small profile, one epoch, one GPU.
PROFILE=smoke_30 EPOCHS=1 GPUS=0 bash scripts/diagnosis/run_ctfm_diagnosis.sh

# Full diagnosis: 100-epoch budget, early stop after 15 validation stalls, two GPUs
# (the defaults; batch size 4 comes from the ct_fm_frozen_*.yaml configs, BATCH_SIZE=N overrides).
PROFILE=full_inspect EPOCHS=100 EARLY_STOPPING=15 GPUS=0,1 \
  bash scripts/diagnosis/run_ctfm_diagnosis.sh

# Prognosis variants.
PROFILE=full_inspect GPUS=0,1 bash scripts/prognosis/run_ctfm_prognosis_all.sh
PROFILE=full_inspect TARGET=12_month_PH GPUS=0,1 \
  bash scripts/prognosis/run_ctfm_prognosis_pe.sh
```

`ACTION` for these wrappers is `prepare`, `train`, `evaluate`, `all`, `preflight`, or
`dry`. `all` prepares the CT-FM cache, trains the MLP, then evaluates the untouched test
split. The task output is intentionally compact: one `epoch_<EPOCHS>/` folder per training
budget, so `EPOCHS=1` and `EPOCHS=30` of the same run sit side by side and only repeating a
budget is an output collision (`OVERWRITE=1` replaces just that folder). Each folder holds
`resolved_config.yaml`, `result.json`, `checkpoint/{best.ckpt,last.ckpt}`, `logs.txt`,
`result.csv`, `predictions.csv`, `training_curves.png`, and a validation-only `preview/` for up
to five patients: per patient an offline `.html` Grad-CAM viewer (every input slice as CT |
CT + CAM, opacity, 8-slice top-CAM montage) and a `.png` summary; see
`docs/05_diagnosis_training.md` "Preview Grad-CAM". `python tools/tasks/gradcam_preview.py
--run-dir <run>` rebuilds it for an evaluated run without re-evaluating.
`result.csv` has one row per target and split (`train`, `validation`, `test`) with the same
columns in every run, including zero-shot PENet (`n_pos`/`n_neg`, AUROC/AUPRC, the
validation-selected `threshold` and its `threshold_rule`, threshold metrics, and a `note`
explaining every empty cell); only the test row carries patient-bootstrap CIs. For prognosis
this includes all seven endpoints plus calibration columns. `logs.txt` holds the training log
followed by the evaluation section (thresholds, per-split case counts, warnings, final test
block); launcher command lines are not stored there. Column meanings:
`docs/09_output_reference.md`.
Diagnosis reports AUROC/AUPRC, sensitivity, specificity, F1, Brier and
the validation-selected threshold; prognosis reports the same plus calibration metrics and
a calibration curve. Prognosis runs default to all seven endpoints; set `TARGET` to run
one endpoint independently, with a separate output ID. Early stopping selects the best
checkpoint using negative validation loss; the final clinical metrics are computed only
afterward by `evaluate.py` on the test split. CT preprocessing QC counts are in
`data_quality.md` and the CT-FM cache's `dataset.json`; failure ID CSVs appear only
when cases fail. QC is not mixed into task clinical metrics.

Progress on long runs: `prepare` prints `N/total computed (s/study, ~left, failed)` every 25
studies (into `<derived>/datasets/<profile>/logs.txt`). Training and evaluation log a line at
most once a minute per pass (`epoch 3/50 train 2410/18900 batches (…, ~8.8 min left)
loss=…`, the same for `validation`, `epoch AUROC pass train|validation` and `evaluate
train|validation|test`) plus a `run progress` line per epoch with an upper bound for the
remaining epochs; passes shorter than a minute stay silent, so smoke logs look as before.
`PE_PROGRESS_EVERY_SEC=30` changes the interval. While training runs, follow
`epoch_<EPOCHS>/logs/run.log` (it becomes `logs.txt` when training ends; evaluation then
appends to `logs.txt`). `training.record_epoch_auc: true` (the CT-FM default) adds an
inference pass over train and validation every epoch for the AUROC curves; with pooled
inputs (below) that pass is cheap, with `FEATURE_INPUT=grid` it roughly doubles epoch time.

Workers adapt to the machine. `WORKERS` (prepare's preprocessing processes) and
`NUM_WORKERS` (train/evaluate DataLoader workers) default to `auto`: sized from the CPU count
and the RAM free at start (`source/utils/workers.py`; measured budgets 2 GB per
preprocessing worker, 0.5 GB per loader worker, 4 GB kept for the main process). A number
is an upper bound: when free memory cannot hold it the run uses fewer and logs why, e.g.
`workers=5 (requested 12, capped to avoid OOM: 4 CPUs, 6.6 GB free RAM, …)`. The same command
therefore fits the 4-vCPU/15 GB VM (≈3 preprocessing workers when idle) and a 16-vCPU/64 GB
one (≈15). Evaluation now also loads with workers (it used none).

Pooled features. `prepare` writes, next to each `features/<study>.npy` grid, a pooled copy
`pooled/<study>.npy` (float32 `[513, 1, 1, 1]` + sidecar): the body-weighted mean the MLP
takes from the grid, plus a weight channel of 1, so the unchanged model reads it as a
one-cell grid. `manifests/ct_fm/*.csv` gain a `pooled_path` column, and the three frozen
arms (`ct_fm_frozen_{diagnosis,prognosis_all,prognosis_pe}.yaml`) train and evaluate on it
with `data.preload_inputs: true`: all ~2 KB files are read once into RAM instead of
re-reading 5.9 MB per study per pass. Previews still render on the grid and check their
forward pass against the pooled probabilities. Checks on smoke_30: a fixed checkpoint gives
the same logits on grid and pooled inputs (max difference < 1e-6); retraining diagnosis
reproduced every loss and prediction; retraining prognosis drifted by at most 6e-4 in
probability (float rounding amplified by Adam, the same order as changing GPU).
`FEATURE_INPUT=grid` reads the grids as before; the anatomy arms always use the grid. A
cache built before pooling existed gets its pooled copies from `ACTION=prepare` without
recomputing CT-FM (train/evaluate refuse a manifest without `pooled_path` and say so).

## Segmentation, ROI and silver labels

```bash
PROFILE=smoke_30 MAX_CASES=1 GPUS=0 bash scripts/tool/run_segmentation.sh   # one study first
PROFILE=smoke_30 GPUS=0 bash scripts/tool/run_segmentation.sh               # whole profile
PROFILE=smoke_30 bash scripts/tool/run_roi.sh                               # after segmentation
PROFILE=smoke_30 GPUS=0 bash scripts/tool/run_silver_labels.sh
```

These accept `ACTION=run|preflight`, `OVERWRITE=1`,
and one scope: `PATIENT_ID=<id>`, `MAX_CASES=N` (`MAX_REPORTS=N` for silver labels), or the
whole manifest by default. Segmentation writes `masks/<patient>/<study>/<anatomy>.nii.gz`,
one `previews/<patient>_<study>.png` per study and `qc_summary.csv`; see
`docs/02_segmentation.md`. `run_roi.sh` also takes `SEGMENTATION_RUN` and `ROI_WORKERS`.

Generation stages (dataset, segmentation, ROI, silver labels, counterfactual) still need an
explicit scope: pass `--max-cases N`, `--patient-id <id>` or `--allow-full`. A training stage
has no per-case limit — its scope *is* the dataset profile, so rehearse on
`--set data.profile=test_500_sample` and use `preflight` / `dry` to check a run without
starting it.

## Storage

The pipeline writes only under `PE_CLOUD_ROOT`, `/mnt/pe-project/outputs` by default
(`configs/paths.yaml`, same layout as `gs://pe-study/pe-storage`). The only mount is the raw
release: mount `gs://pe-study/Stanford_INSPECT_dataset` at `/mnt/Stanford_INSPECT_dataset`
yourself with gcsfuse — nothing in this repository mounts anything.
Back the outputs up to the bucket with `gcloud storage rsync --recursive /mnt/pe-project/outputs gs://pe-study/pe-storage` (never deletes).

`scripts/tool/use_gcs_storage.sh` sets the roots; values you have already exported win:

| variable | default |
|---|---|
| `PE_CLOUD_ROOT` | `/mnt/pe-project/outputs` |
| `PE_CLOUD_PROJECT_ROOT` | `$PE_CLOUD_ROOT/pe-project` (outputs go to `…/outputs`) |
| `PE_RAW_INSPECT_ROOT` | `/mnt/Stanford_INSPECT_dataset` |
| `PE_DERIVED_ROOT` | `$PE_CLOUD_ROOT/derived` |

Every `scripts/*/*.sh` sources it. To have the same variables for direct `python run.py …`
or `tools/…` calls without exporting them each time, install it into `~/.bashrc` once:

```bash
bash scripts/tool/use_gcs_storage.sh --install     # new terminals get the variables
bash scripts/tool/use_gcs_storage.sh --show        # current values + mount check
bash scripts/tool/use_gcs_storage.sh --uninstall   # remove the ~/.bashrc line
```

It warns when no dataset profile exists under `PE_DERIVED_ROOT` or the raw release is missing.
`PE_LOCAL_CACHE_ROOT` (optional) moves the local code-side cache, default `<repo>/cache`.

## Zero-shot PENet

The released PENet is a trained CTPA PE classifier, so applying it unchanged is an
external baseline rather than another arm to train.

```bash
pip install opencv-python-headless        # cv2.INTER_AREA slice resize, required

CHECK_SLICE_ORDER=1 bash scripts/diagnosis/run_zeroshot_penet.sh   # run this first
SLICE_ORDER=<winner> bash scripts/diagnosis/run_zeroshot_penet.sh  # then the full test split
```

Run the check first. A 3-D convolution is sensitive to slice order inside a window and
the PENet repository never records which direction its pre-sorted volumes used, so the
check scores 50 cases both ways; the wrong direction lands near chance. Getting this
wrong produces a plausible-looking but meaningless baseline with no error message.

`PROFILE`, `MAX_CASES`, `RESTRICT_TO`, `AGGREGATE` (`max`/`mean`), `GPUS`, `CHECKPOINT`
and `OVERWRITE` are the other knobs. Pass `RESTRICT_TO` whenever the row has to sit in
the same table as an arm that is not computable for every case.

Output: `outputs/diagnosis/DX_zeroshot_penet__ds_<profile>/` with `result.csv` (validation and
test rows, same columns as the CT-FM `result.csv`, so the tables can be stacked and compared),
`predictions.csv`, `logs.txt`, `result.json` and `resolved_config.yaml`. An existing run is
never overwritten silently; re-run with `OVERWRITE=1`. `logs.txt` gets a
`scored N/total (s/study, ~h left)` line every 25 studies and one line per skipped study;
follow it with `tail -f`. The run cannot resume: an interruption starts it over.

## Zero-shot RADAR

RADAR (`third_party/repos/damo-radar`, DAMO's vision-language generalist for abdominal CT)
ships no PE finding. The run scores RADAR's own pulmonary-artery organ token against the text
pair in `configs/runs/01_foundation/radar_zero_shot.yaml` (`radar.prompts`, written for this
project), so it is an out-of-scope probe: RADAR was trained on portal-venous abdominal CT,
resamples to 5 mm slices and clips at 400 HU.

```bash
# once: RADAR pins transformers==4.25, so it gets its own env
conda create -n radar python=3.10 && conda activate radar
pip install -r third_party/repos/damo-radar/requirements.txt
conda activate pe

PROFILE=smoke_30 MAX_CASES=5 bash scripts/diagnosis/run_zeroshot_radar.sh   # quick look
bash scripts/diagnosis/run_zeroshot_radar.sh                                # full validation + test
RESUME=1 bash scripts/diagnosis/run_zeroshot_radar.sh                       # continue after an interruption
```

`PYTHON` (project env) selects studies and evaluates; `RADAR_PYTHON` (default
`<conda base>/envs/radar/bin/python`) runs `tools/tasks/zeroshot_radar_worker.py`, which
reproduces `RADAR_inference/inference_demo.py` per study. `GPUS` defaults to `0` (the model is
impractical on CPU). Scores are appended per study, so `RESUME=1` continues where a run
stopped; `OVERWRITE=1` starts clean. `PROFILE`, `MAX_CASES`, `RESTRICT_TO` and `CHECKPOINT` work
as for PENet. About 5 s per study on one L4.

Studies where RADAR's segmentation finds no pulmonary artery (for example a chest-abdomen-
pelvis series that starts below it) cannot be scored; they are listed under
`skipped_series` in `result.json` and left out of the metrics, so compare against another arm
on the common studies. `radar | RuntimeError: module compiled against ABI version ...` in the
log comes from the pinned `opencv-python-headless==4.5.5.64` under numpy 2; MONAI imports cv2
optionally and the RADAR path never uses it.

Output: `outputs/diagnosis/DX_zeroshot_radar__ds_<profile>/` with the same `result.csv`,
`predictions.csv` (plus `windows`, `organ_voxels`, `fallback_crop`), `logs.txt`,
`result.json` (input contract, prompts, repo commit) and `preview/` (first five validation
studies: RADAR's input with its pulmonary-artery mask, TP/TN/FP/FN in the file name).

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
