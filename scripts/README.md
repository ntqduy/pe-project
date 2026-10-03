# `scripts/`

The general launcher is still `run.py`; the baseline experiments and data stages also have
explicit wrappers. The wrappers only select a reviewed config/cohort; preprocessing and model
logic remain in `source/` and `tools/`.

```text
scripts/
├── data/                       stage 0: build and describe the data artifacts
│   ├── preprocessing.sh        stage 0 end to end for one dataset profile
│   ├── segmentation.sh         TotalSegmentator + LungMask masks, contact-sheet previews, QC
│   ├── roi.sh                  ROI1-ROI8 and matched random controls
│   ├── silver_labels.sh        report-derived accepted/abstained silver labels (MedGemma)
│   └── eda.sh                  EDA over a built dataset profile (see analysis/README.md)
├── diagnosis/
│   ├── zero_shot/
│   │   ├── penet.sh            released PENet applied unchanged, no training
│   │   └── radar.sh            released RADAR (abdominal-CT generalist) probed for PE, no training
│   └── baselines/              the 2D / 2.5D / 3D baseline zoo (see baselines/README.md)
│       ├── exp01_baselines/{2D,2_5D,3D}/   20 models, MLP head; one wrapper per model + run_all.sh
│       ├── exp02_data_fraction/frac0NN/    training-set size: one folder per 25/50/75/100% (subsets per seed)
│       ├── exp03_head_ablation/            <model>_kan.sh: KAN head (MLP arm = exp01 run)
│       ├── exp04_slice_ablation/{2D,2_5D}/ attention-MIL vs mean / max pooling vs middle slice
│       ├── prepare_ctfm_cache.sh  CT-FM feature cache, once per profile (ctfm_frozen_3d)
│       ├── prepare_weights.sh  fetch the pretrained weights of every arm once
│       ├── smoke.sh            one bf16 train step per arm on GPU
│       └── summarize.sh        rebuild the tables / plots from finished runs
├── prognosis/
│   └── baselines/              exp01..exp04 for prognosis (--label required), same grids
└── tool/                       shared code only; nothing here is one experiment
    ├── use_gcs_storage.sh      storage roots (PE_*); sourced by every script, see Storage
    ├── _flags.sh               is_true / is_false for on/off environment flags
    └── run_baseline_grid.sh    env vars -> tools/baselines/run_many.py, behind every baselines wrapper
```

## Baseline experiments (quick reference)

Every experiment has one folder with its grid (`experiment.yaml`) and a `run_all.sh`; per-model
wrappers run one model of it. All take the same flags (or the environment variables listed in
`tool/run_baseline_grid.sh`); cases run in parallel over `--gpus`, finished cases are skipped.

```bash
E=scripts/diagnosis/baselines
bash $E/exp01_baselines/run_all.sh --gpus 0,1 --seeds "0 1 2"            # 20 models, MLP
bash $E/exp02_data_fraction/run_all.sh --gpus 0 --runs-per-gpu 2          # 25/50/75/100%
bash $E/exp02_data_fraction/frac025/run_all.sh --gpus 0                   # 25% only
bash $E/exp03_head_ablation/run_all.sh --gpus 0                           # KAN (MLP from exp01)
bash $E/exp04_slice_ablation/run_all.sh --gpus 0 --variants "center mean" # 2D/2.5D slice ablation
bash $E/exp01_baselines/3D/vit_3d.sh --gpus 1                             # one model
bash scripts/prognosis/baselines/exp01_baselines.sh --label 12_month_PH --gpus 0   # prognosis
bash $E/exp01_baselines/run_all.sh --dry-run                              # list cases + commands only
bash $E/summarize.sh exp01_baselines                                      # rebuild the tables
python tools/baselines/smoke_pipeline.py                                  # pipeline check, synthetic data
```

| flag | meaning |
|---|---|
| `--task diagnosis\|prognosis`, `--label <outcome>`, `--cohort all\|pe` | prognosis needs one of the 7 outcomes; diagnosis rejects `--label` |
| `--seeds "0 1 2"` | official split, repeated per seed (default `0 1 2`) |
| `--gpus 0,1`, `--runs-per-gpu N` | GPU pool and cases per GPU (`''` = CPU) |
| `--variants`, `--heads`, `--fractions` | subset of the experiment's grid |
| `--dry-run` | print every case and its `run_case.py` command, start nothing |
| anything else | passed to every `tools/baselines/run_case.py` (e.g. `--overwrite`, `--set key=value`) |

**Add a seed:** run again with the extra seed (`--seeds "3"`); runs already finished are kept,
and `summarize.sh` pools every seed it finds (`n_seeds` column, seed-ensemble CI).

**Add a model:** (1) put the encoder in `source/model/2D_model/` or `source/model/3D_model/`
and register its builder in `source/model/registry.py`; (2) add its contract (feature_dim,
pretrained weights, intensity) to `configs/components/backbones.yaml#registry`; (3) copy a run
config into `configs/runs/02_diagnosis/baselines/<2D|2_5D|3D>/<model>.yaml` (micro-batch so the
effective batch stays 4); (4) add it to `MODEL_GROUPS` in `tools/baselines/experiments.py` and
to the `models:` list of each `experiment.yaml` that should include it, plus a one-line wrapper
like its neighbours; (5) check it with `python tools/baselines/smoke.py --models <model>` and
`python tools/baselines/smoke_pipeline.py --models <model> --cases-only --seeds 0`.

**Add an experiment:** a new `scripts/diagnosis/baselines/expNN_<name>/experiment.yaml`
(`title`, `models`, `heads`, `fractions`, optional `variants` / `overrides` / `split_seed`),
its name in `EXPERIMENT_NAMES` (`tools/baselines/experiments.py`), and a `run_all.sh`.

What goes where: a wrapper that builds or describes a data artifact lives in `data/`; a
wrapper for one diagnosis or prognosis arm lives under `diagnosis/` or `prognosis/`, in a
subfolder named for its family (`zero_shot/`, `baselines/`); `tool/` holds only
code shared by several wrappers and is not an experiment by itself. File names say what runs,
not how (`penet.sh`, not `run_zeroshot_penet.sh`): the folder already gives the stage and family.

Every script takes its settings from environment variables in front of the command
(`PROFILE=smoke_30 GPUS=0 bash scripts/<folder>/<name>.sh`). Common ones: `PROFILE`
(`smoke_30` | `test_500_sample` | `full_inspect`, default `full_inspect`; `eda.sh` defaults
to `test_500_sample`), `ACTION`, `GPUS`, `OVERWRITE=1`, `PYTHON`. On/off flags (`OVERWRITE`,
`RESUME`, `REBUILD_CACHE`, `VERIFY_CACHE`, `SMOKE`, `SCRATCH`, `DRY_LIST`, `NO_FIGURES`,
`CHECK_SLICE_ORDER`, ...) are on for `1`, `true`, `yes` or `on` (any case) and off when
unset, empty, `0`, `false`, `no` or `off`; any other value (`OVERWRITE=maybe`) stops the
script with an error (`scripts/tool/_flags.sh`). The two default-on switches, the baseline
grid's `EPOCH_AUC` and segmentation's `LUNGMASK`, are turned off with `=0` (or `false`/`no`/`off`).

## Building a dataset profile

```bash
PROFILE=smoke_30 bash scripts/data/preprocessing.sh        # rehearse first
ACTION=preflight bash scripts/data/preprocessing.sh        # check the full build
bash scripts/data/preprocessing.sh                         # full_inspect, hours
```

`ACTION` picks `run`, `preflight`, `plan` or `show` (`run.py` refuses `dry` for data stages,
so the wrapper rejects it up front); `OVERWRITE=1` rebuilds over an
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
python run.py show  baseline.resnet18_3d # what it is, what it needs, what it produces
python run.py plan  baseline.resnet18_3d # dependency chain: READY / MISSING / BLOCKED
python run.py preflight baseline.resnet18_3d --gpus 0
python run.py dry   baseline.resnet18_3d --gpus 0
python run.py run   baseline.resnet18_3d --gpus 0   # train one run; grids: tools/baselines/run_case.py
```

## What the old environment variables map to

| old wrapper variable | now |
|---|---|
| `DATASET=<profile>` | `--set data.profile=<profile>` |
| `BACKBONE=<name>` | `--set model.backbone=<name>` |
| `SMOKE=1` | `--set training.epochs=1` |
| `GPUS=0,1` | `--gpus 0,1` |
| `ACTION=<verb>` | the `run.py` subcommand (`show`, `plan`, `preflight`, `dry`, `run`) |
| `ALLOW_ALL=1` | `--allow-full` |
| `MAX_CASES=N` / `MAX_REPORTS=N` / `PATIENT_ID=id` | `--max-cases N` / `--max-reports N` / `--patient-id id` |
| `RESUME=1` / `OVERWRITE=1` | `--resume` / `--overwrite` |
| `REBUILD_CACHE=1` (`prepare_ctfm_cache.sh`) | `build_ctfm_cache.py --overwrite`; `OVERWRITE=1` never rebuilds the feature cache |
| `SET="a=1 b=2"` | `--set a=1 --set b=2` |

A baseline case (`tools/baselines/run_case.py`, behind every `exp0*` wrapper) writes one
`epoch_<E>/` folder per training budget under
`<outputs>/<family>/BASE/<profile>/<task>/runs/<model>__<head>__frac<PPP>[__v<variant>]/official_seed<S>/`,
so `EPOCHS=1` and `EPOCHS=100` of the same case sit side by side (`OVERWRITE=1` replaces just
that folder). Each folder holds `resolved_config.yaml`, `result.json`,
`checkpoint/{best.ckpt,last.ckpt}`, `logs.txt`, `result.csv`, `predictions.csv`,
`training_curves.png` and `visualize/{correct,incorrect}/`: Grad-CAM of the 3 most confident
correct and 3 most confident wrong **test** cases, per case an offline `.html` viewer (every
input slice as CT | CT + CAM, opacity, top-CAM montage), a `.png` summary and NIfTI
`_ct.nii.gz` / `_gradcam.nii.gz`; see `docs/03_training_evaluation.md` (Grad-CAM preview).
`ACTION=evaluate` rebuilds it. `result.csv` has one row per target and split (`train`,
`validation`, `test`) with the same columns in every run, including zero-shot PENet
(`n_pos`/`n_neg`, AUROC/AUPRC, the validation-selected `threshold` and its `threshold_rule`,
threshold metrics, and a `note` explaining every empty cell); only the test row carries
patient-bootstrap CIs. Prognosis adds calibration columns. `logs.txt` holds the training log
followed by the evaluation section (thresholds, per-split case counts, warnings, final test
block); launcher command lines are not stored there. Column meanings:
`docs/05_running_outputs.md`. `best.ckpt` is the epoch with the highest validation AUROC; the
final clinical metrics are computed only afterward by `evaluate.py` on the test split. CT
preprocessing QC counts are in `data_quality.md` and the CT-FM cache's `dataset.json`;
failure ID CSVs appear only when cases fail. QC is not mixed into task clinical metrics.

Progress on long runs: `prepare_ctfm_cache.sh` prints `N/total computed (s/study, ~left, failed)` every 25
studies (into `<derived>/datasets/<profile>/logs.txt`). Training and evaluation log a line at
most once a minute per pass (`epoch 3/50 train 2410/18900 batches (…, ~8.8 min left)
loss=…`, the same for `validation`, `epoch AUROC pass train|validation` and `evaluate
train|validation|test`) plus a `run progress` line per epoch with an upper bound for the
remaining epochs; passes shorter than a minute stay silent, so smoke logs look as before.
`PE_PROGRESS_EVERY_SEC=30` changes the interval. While training runs, follow
`epoch_<EPOCHS>/logs/run.log` (it becomes `logs.txt` when training ends; evaluation then
appends to `logs.txt`). `training.record_epoch_auc: true` adds an inference pass over train
and validation every epoch for the AUROC curves (`EPOCH_AUC=0` skips the train pass).

Workers adapt to the machine. `WORKERS` (the CT-FM cache's preprocessing processes) and
`NUM_WORKERS` (train/evaluate DataLoader workers) default to `auto`: sized from the CPU count
and the RAM free at start (`source/utils/workers.py`; measured budgets 2 GB per
preprocessing worker, 0.5 GB per loader worker, 4 GB kept for the main process). A number
is an upper bound: when free memory cannot hold it the run uses fewer and logs why, e.g.
`workers=5 (requested 12, capped to avoid OOM: 4 CPUs, 6.6 GB free RAM, …)`. The same command
therefore fits the 4-vCPU/15 GB VM (≈3 preprocessing workers when idle) and a 16-vCPU/64 GB
one (≈15). Evaluation now also loads with workers (it used none).

## CT-FM feature cache (`ctfm_frozen_3d`)

```bash
bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh                         # full_inspect, cuda:0
PROFILE=smoke_30 GPUS='' bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh  # CPU
VERIFY_CACHE=1 bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh          # re-check every cached study
bash scripts/diagnosis/baselines/exp01_baselines/3D/ctfm_frozen_3d.sh          # then the arm itself
```

Run it once per dataset profile before any `ctfm_frozen_3d` case. It wraps
`tools/data/build_ctfm_cache.py`, never creates a split (it reads the profile's
official-split manifests), and writes per-study `features/<study>.npy` grids plus pooled
copies `pooled/<study>.npy` (float32 `[513, 1, 1, 1]`: the body-weighted mean plus a weight
channel) and `manifests/ct_fm/*.csv` with a `pooled_path` column. `ctfm_frozen_3d` trains
and evaluates on the pooled copies with `data.preload_inputs: true`. A finished cache is
recognised in seconds; an unfinished one resumes. `REBUILD_CACHE=1` recomputes every study
(~20 h on `full_inspect`); `WORKERS` sizes the preprocessing processes. Progress goes to
`<derived>/datasets/<profile>/logs.txt`.

## Segmentation, ROI and silver labels

```bash
PROFILE=smoke_30 MAX_CASES=1 GPUS=0 bash scripts/data/segmentation.sh   # one study first
PROFILE=smoke_30 GPUS=0 bash scripts/data/segmentation.sh               # whole profile
PROFILE=smoke_30 bash scripts/data/roi.sh                               # after segmentation
PROFILE=smoke_30 GPUS=0 bash scripts/data/silver_labels.sh
```

These accept `ACTION=run|preflight`, `OVERWRITE=1`,
and one scope: `PATIENT_ID=<id>`, `MAX_CASES=N` (`MAX_REPORTS=N` for silver labels), or the
whole manifest by default. Segmentation writes `masks/<patient>/<study>/<anatomy>.nii.gz`,
one `previews/<patient>_<study>.png` per study and `qc_summary.csv`; see
`docs/01_data_pipeline.md`. `roi.sh` also takes `SEGMENTATION_RUN` and `ROI_WORKERS`.
Segmentation and ROI continue a partial run by default, but refuse to resume when an
output-affecting setting changed since the run started (recorded in the run's
`resume_settings.json`; the error names the changed keys): rerun with `OVERWRITE=1`, restore
the settings, or use another experiment id.

Generation stages (dataset, segmentation, ROI, silver labels, zero-shot) still need an
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

Every script under `scripts/` sources it, directly or through its shared driver in
`scripts/tool/`. To have the same variables for direct `python run.py …`
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

CHECK_SLICE_ORDER=1 bash scripts/diagnosis/zero_shot/penet.sh   # run this first
SLICE_ORDER=<winner> bash scripts/diagnosis/zero_shot/penet.sh  # then the full test split
```

Run the check first. A 3-D convolution is sensitive to slice order inside a window and
the PENet repository never records which direction its pre-sorted volumes used, so the
check scores 50 cases both ways; the wrong direction lands near chance. Getting this
wrong produces a plausible-looking but meaningless baseline with no error message.

`PROFILE`, `MAX_CASES`, `RESTRICT_TO`, `AGGREGATE` (`max`/`mean`), `GPUS`, `CHECKPOINT`
and `OVERWRITE` are the other knobs. Pass `RESTRICT_TO` whenever the row has to sit in
the same table as an arm that is not computable for every case.

Output: `outputs/diagnosis/DX_zeroshot_penet__ds_<profile>/` with `result.csv` (validation and
test rows, same columns as every trained arm's `result.csv`, so the tables can be stacked and compared),
`predictions.csv`, `logs.txt`, `result.json` and `resolved_config.yaml`. An existing run is
never overwritten silently; re-run with `OVERWRITE=1`. `logs.txt` gets a
`scored N/total (s/study, ~h left)` line every 25 studies and one line per skipped study;
follow it with `tail -f`. The run cannot resume: an interruption starts it over.

## Zero-shot RADAR

RADAR (`third_party/repos/damo-radar`, DAMO's vision-language generalist for abdominal CT)
ships no PE finding. The run scores RADAR's own pulmonary-artery organ token against the text
pair in `configs/runs/01_foundation/zero_shot/radar.yaml` (`radar.prompts`, written for this
project), so it is an out-of-scope probe: RADAR was trained on portal-venous abdominal CT,
resamples to 5 mm slices and clips at 400 HU.

```bash
# once: RADAR pins transformers==4.25, so it gets its own env
conda create -n radar python=3.10 && conda activate radar
pip install -r third_party/repos/damo-radar/requirements.txt
conda activate pe

PROFILE=smoke_30 MAX_CASES=5 bash scripts/diagnosis/zero_shot/radar.sh   # quick look
bash scripts/diagnosis/zero_shot/radar.sh                                # full validation + test
RESUME=1 bash scripts/diagnosis/zero_shot/radar.sh                       # continue after an interruption
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

Sweeps are loops over flags, not separate code paths. A prognosis outcome sweep:

```bash
for L in 1_month_mortality 12_month_mortality 12_month_PH; do
  bash scripts/prognosis/baselines/exp01_baselines.sh --label "$L" --cohort pe --gpus 0,1
done
```

`run_case.py` changes the manifest, `data.cohort`, `data.label_columns`, `task.primary_target`
and `task.targets` together, and each (cohort, outcome) gets its own
`prognosis_<cohort>_<label>/` folder.

See [docs/04_experiments.md](../docs/04_experiments.md) for the full result-table recipes.
