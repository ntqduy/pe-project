# Baseline experiments (2D / 2.5D / 3D)

Four experiments, one folder each, one shared driver; every one runs for diagnosis and, with
`--task prognosis --label <outcome>` (or `scripts/prognosis/baselines/<exp>.sh`), for prognosis.
`experiment.yaml` in each folder is the case grid (models, heads, training fractions, optional
per-model `overrides`, optional `variants`, optional `split_seed: per_seed`); each model's own
contract is `configs/runs/02_diagnosis/baselines/<2D|2_5D|3D>/<model>.yaml`. A model given
overrides in one experiment gets separate runs tagged `__v<experiment>`; a named variant gets
runs tagged `__v<variant>` (exp04: `center`, `mean`, `max`; `default` is the shared exp01 run).
Quick reference of the flags: [scripts/README.md](../../README.md#baseline-experiments-quick-reference).

```text
scripts/
├── diagnosis/baselines/
│   ├── exp01_baselines/       20 arms x MLP head x 100% train      experiment.yaml + run_all.sh + {2D,2_5D,3D}/<model>.sh
│   ├── exp02_data_fraction/    8 arms x MLP head x 25/50/75/100%   experiment.yaml + run_all.sh + 3D/<model>.sh
│   ├── exp03_head_ablation/    6 arms x {MLP, KAN} x 100%          experiment.yaml + run_all.sh + 3D/<model>.sh
│   ├── exp04_slice_ablation/   ResNet-18 2D/2.5D x {attention, mean, max, center}   + {2D,2_5D}/<model>.sh
│   ├── prepare_weights.sh     download every pretrained weight once + weight-status table
│   ├── smoke.sh               one bf16 train step per arm at 128^3 on GPU (grads, CAM, peak VRAM)
│   └── summarize.sh           rebuild tables / plots from finished runs
└── tool/
    └── run_baseline_grid.sh   env vars -> tools/baselines/run_many.py (shared by every wrapper above)
```

## Quick start (VM)

```bash
# 0. once: the Mamba arms need the CUDA mamba_ssm; the Mamba-MAE fork serves all three
pip install -e third_party/repos/mamba_mae/causal-conv1d
pip install -e third_party/repos/mamba_mae/mamba2
pip install fvcore                       # imported by third_party/repos/VMamba/vmamba.py

# 1. fetch weights once (before parallel runs) and check what loads
bash scripts/diagnosis/baselines/prepare_weights.sh

# 1b. one bf16 step per arm at the real size: catches CUDA-only errors and OOM early
bash scripts/diagnosis/baselines/smoke.sh

# 1c. whole pipeline on synthetic data (no INSPECT data, CPU is enough; writes under output/_smoke)
python tools/baselines/smoke_pipeline.py                  # report: output/_smoke/smoke_report.md
python tools/baselines/run_case.py --model vit_3d --head kan --smoke   # one case

# 2. rehearse one arm on the small profile, then the real thing
PROFILE=smoke_30 EPOCHS=1 bash scripts/diagnosis/baselines/exp01_baselines/3D/resnet18_3d.sh
bash scripts/diagnosis/baselines/exp01_baselines/3D/resnet18_3d.sh

# 3. a whole experiment over 4 GPUs (one case per GPU; finished cases are skipped)
GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp01_baselines/run_all.sh
GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh
GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp03_head_ablation/run_all.sh
```

A CT-FM frozen arm needs the CT-FM feature cache first (`ACTION=prepare bash
scripts/tool/run_ctfm_frozen.sh`), exactly like the existing CT-FM runs.

## Knobs (environment variables, see `scripts/tool/run_baseline_grid.sh`)

| variable | default | meaning |
|---|---|---|
| `PROFILE` | `full_inspect` | dataset profile |
| `GPUS` | `0` | GPU pool; `''` = CPU |
| `JOBS_PER_GPU` | `1` | cases sharing one GPU (2-4 for `ctfm_frozen_3d`) |
| `GPUS_PER_JOB` | `1` | `>1` = one DDP case over several GPUs |
| `FOLDS` | `official` | `official` and/or fold numbers: `FOLDS="0 1 2 3 4"`; `K=5` |
| `SEEDS` | `42` | `SEEDS="42 43 44"` for repeated runs |
| `HEADS`, `FRACTIONS` | experiment's | e.g. `HEADS=kan`, `FRACTIONS="25 50"` |
| `ACTION` | `all` | `all`, `prepare`, `train`, `evaluate`, `preflight`, `dry` |
| `EPOCHS`, `EARLY_STOPPING`, `BATCH_SIZE`, `ACCUMULATION`, `LR` | config | training overrides |
| `EPOCH_AUC=0` | on | skip the per-epoch AUROC pass (halves epoch time) |
| `SCRATCH=1` | off | ignore pretrained weights |
| `OVERWRITE=1` | off | replace finished cases |
| `TASK=prognosis` | `diagnosis` | image-only prognosis; `LABEL` is then required (`1_month_mortality`, `6_month_mortality`, `12_month_mortality`, `1_month_readmission`, `6_month_readmission`, `12_month_readmission`, `12_month_PH`); `COHORT=all` or `pe`. `LABEL` with diagnosis is an error |
| `DRY_LIST=1` | off | print the case grid only |

One case directly: `python tools/baselines/run_case.py --model vit_3d --head kan --fraction 50 --seed 1 --gpus 1`
(prognosis: `--task prognosis --label 12_month_PH`).

## Training protocol (every arm, `configs/components/baselines.yaml`)

| setting | value |
|---|---|
| split | official INSPECT train / validation / test; repeat with seeds (`SEEDS="0 1 2"`), not folds |
| effective batch | 4 = micro-batch x `gradient_accumulation`; an arm that does not fit lowers its micro-batch in its run config (`vmamba_3d`, `mamba_mae_3d`: 1 x 4; most 2D/3D arms: 2 x 2) |
| epochs / early stopping | at most 100; stop after 15 epochs without a better validation AUROC |
| checkpoint | `best.ckpt` = epoch with the highest validation AUROC (`training.selection_metric: val_auroc`) |
| precision | `compute.precision: auto` = bf16 if the GPU supports it, else fp16 + GradScaler, fp32 on CPU |
| memory | per-arm `gradient_checkpointing` in `backbones.yaml`; `run.log` prints device, precision, micro / effective batch and checkpointing |
| seeds | python / numpy / torch / CUDA, cuDNN deterministic, seeded DataLoader order and workers |
| threshold | Youden on validation, applied unchanged to test |
| CI | 95% patient-level bootstrap (2000 resamples) on test |

## What a case is

`encoder -> shared projection (Dropout, Linear(F, 64), LayerNorm) -> head (MLP | KAN)`,
built by `source/model/classifier.py` (`task.architecture: baseline_classifier`). Encoders
live in `source/model/2D_model` and `source/model/3D_model`, heads in `source/model/head`;
their contracts are in `configs/components/backbones.yaml`, shared task / head / budget
presets in `configs/components/baselines.yaml`, one run config per arm in
`configs/runs/02_diagnosis/baselines/`.

| input | how |
|---|---|
| 3D | the cached 128^3 RAS volume (1.5 mm) |
| 2D | 32 axial slices at uniform z, each through the 2D backbone, gated-attention MIL over slices |
| 2.5D | as 2D, each instance = 3 adjacent slices as the RGB channels |
| center ablation | `--set model.mil.slice_selection=center`: 2D = the middle slice only, 2.5D = the 3 middle slices |

Each encoder maps the cache's `[0,1]` of `[-1000, 1000]` HU to its own pre-training
convention (ImageNet window + normalisation, MedicalNet / Mamba-MAE z-score, Swin-UNETR
unit scaling, PENet window).

## Pretrained weights (logged at every build)

Every training log has one line per run, e.g.

```text
PRETRAINED LOADED | ResNet-18 3D | source=huggingface:TencentMedicalNet/MedicalNet-Resnet18 (23 datasets) | matched 102/102 tensors
NO PRETRAINED - training from scratch | nnMamba4cls 3D | reason: upstream nnMamba publishes no classification weights
```

plus `pretrained_weights=` (JSON: matched / missing / shape-mismatched keys) in the log and
in `result.json`. A load that matches under 50% of the tensors is an error, not a silent
partial load. Every arm with published weights has `pretrained.required: true`, so a failed
download stops that case instead of quietly training it from scratch; `SCRATCH=1` is the
explicit from-scratch switch (also for CT-FM LoRA). The matched / missing counts are the real
`load_state_dict` results (for timm models the load timm performs internally is observed).

| arm | weights | adaptation |
|---|---|---|
| ResNet-18/50 3D | MedicalNet (23 CT/MRI datasets) | none |
| DenseNet-121 3D | ImageNet (torchvision) | 2D -> 3D inflation |
| ConvNeXt-T 3D | ImageNet-22k -> 1k | 2D -> 3D inflation |
| ViT-S/16 3D | ImageNet-21k -> 1k | patch kernel inflated, positions resized + repeated along depth |
| Swin 3D | Swin-UNETR self-supervised on ~5k CT | none (`mlp.fc -> mlp.linear` rename) |
| nnMamba4cls 3D | none published | from scratch |
| Mamba-MAE 3D | MAE on BraTS MRI (`third_party/weights/Mamba_MAE.pth`) | 4 MRI channels summed, positions 40^3 -> 32^3 |
| VMamba-B 3D | ImageNet (`vssm_base_VMamba.pth`) | 2D -> 3D inflation, 4 SS2D scan params reused by the 4 SS3D scans |
| PENet 3D | released PE model (`penet_best.pth.tar`, SHA-256 checked) | strict load; classifier replaced |
| CT-FM LoRA / frozen | CT-FM feature extractor | LoRA on the two deepest stages / cached features |
| 2D / 2.5D arms | ImageNet (timm) | timm adapts the stem to 1 / 3 channels |

Weight loading is intentionally not assumed from a repository name: each actual run logs the
matched / missing / shape-mismatched tensor counts in `logs.txt` and `result.json`.
Run `prepare_weights.sh` and a per-arm preflight or smoke run in the target environment
before calling a row pretrained; nnMamba has no published classification weights.

## Training fractions (exp02) and k-fold

exp02 has `split_seed: per_seed`: seed `s` draws its own nested subsets, and every fraction
case exports them, with a check, to

```text
<outputs>/<family>/BASE/<profile>/<task>/splits/data_fraction/seed_<s>/
    frac_025.csv  frac_050.csv  frac_075.csv  frac_100.csv     patient_id, study_id, label
    check.json    25 c 50 c 75 c 100 (patients and studies), only official-train IDs, whole
                  patients, positive rate within 0.05 of the full train split -> PASS / FAIL
```

A FAIL stops the case before training. `python tools/baselines/fractions.py --dir <seed dir>`
re-checks a folder. The 100% case is the official split itself (shared with exp01 / exp03).
K-fold stays available (`FOLDS="0 1 2 3 4"`) but the protocol uses the official split + seeds.


`source/data/experiment_splits.py` (called automatically by each case) writes one
patient-level assignment per task and derives manifests under
`<dataset>/manifests/experiments/<task>_<label>_k<K>_s<seed>/<official|fold{k}>/frac<PPP>/`:

- the official **test split never changes**;
- fold k: validation = the pool (official train + validation) patients of fold k, stratified by label;
- fraction p: the train patients whose per-label rank is < p, so fractions are **stratified
  and nested** (25% c 50% c 75% c 100%); each folder also has `train_patients.csv`.

`--fraction` (and `FRACTIONS`) here is always a **whole percent**: `run_case.py --fraction 1`
means 1%, `50` means 50%. `frac<PPP>` is that percent zero-padded (`frac025`, `frac100`).
`tools/data/build_split_manifests.py` parses its `--fraction` differently: a value ending in
`%` or greater than 1 is a percent, one below 1 is a fraction (`0.25`), and a bare `1` is
rejected as ambiguous (write `100` or `1%`). A fractional percent keeps its decimals in the
tag (12.5% -> `frac012p5`), so two fractions never share a folder.

## Outputs

```text
<outputs>/diagnosis/BASE/<profile>/diagnosis/
├── runs/<model>__<head>__frac<PPP>/<fold>_seed<S>/epoch_<E>/     one case (shared by experiments)
│   ├── result.csv  predictions.csv  result.json  logs.txt  training_curves.png
│   ├── checkpoint/{best,last}.ckpt  resolved_config.yaml
│   └── visualize/{correct,incorrect}/  NN_<patient>_<study>_<TP|TN|FP|FN>.{html,png},
│                 _ct.nii.gz + _gradcam.nii.gz (input grid, sidecar affine; ITK-SNAP / 3D Slicer)
│                 (+ _mil_attention.png for 2D / 2.5D)
├── exp01_baselines/      summary.md  summary_pretty.csv  summary_ensemble.csv  summary.csv
│                         summary_raw.csv  auroc_per_model.png  launcher_logs/
│                         runs/<model>_<head>[_frac<PPP>]/<fold>_seed<S> -> symlink to the shared run
├── exp02_data_fraction/  ...                                  auroc_vs_fraction.png
└── exp03_head_ablation/  ...                                  mlp_vs_kan.png
```

A run is identified by its settings, not by the experiment, so e.g. ResNet-18 3D + MLP at
100% is trained once and reused by all three experiments.

`visualize/` explains the **3 most confident correct (TP/TN alternating) and 3 most confident
wrong (FP/FN alternating) test cases** (`preview:` in `baselines.yaml`) at the
validation-selected threshold. The Grad-CAM (HiResCAM) target is each encoder's deepest
feature map: conv stage for CNN/Swin/VMamba/PENet/nnMamba, the last token map reshaped to its
3D grid for ViT / Mamba-MAE, and for 2D / 2.5D the per-slice maps stacked along z; their
montage selects the highest MIL-attention slices. Each viewer's technical table names the
target layer. `python tools/tasks/gradcam_preview.py --run-dir <run>` rebuilds a preview.

Summary tables (`tools/baselines/summarize.py`, also run after every grid):

| file | content |
|---|---|
| `summary.md`, `summary_pretty.csv` | one row per model: AUROC / AUPRC as `0.812 [0.790–0.834]` (seed ensemble) and mean ± std over seeds, every other metric mean ± std, `n_seeds` |
| `summary_ensemble.csv` | the seed ensemble with every `result.csv` column: test probabilities averaged per `study_id` over the seeds (merged on `study_id`, never on row order; runs whose studies or labels differ are an error), threshold re-chosen on the averaged validation probabilities, patient-bootstrap CI |
| `summary.csv` | numeric mean / std / n per metric |
| `summary_raw.csv` | every run's test row (all `result.csv` columns) |

Official-split and k-fold runs are summarised in separate rows (`split` column).
