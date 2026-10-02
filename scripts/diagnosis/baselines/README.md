# Baseline experiments (2D / 2.5D / 3D)

Three experiments, one folder each, one shared driver. `experiment.yaml` in each folder is
the case grid (models, heads, training fractions, optional per-model `overrides`); each
model's own contract is `configs/runs/02_diagnosis/baselines/<2D|2_5D|3D>/<model>.yaml`. A model given overrides in
one experiment gets separate runs tagged `__v<experiment>`.

```text
scripts/
├── diagnosis/baselines/
│   ├── exp01_baselines/       20 arms x MLP head x 100% train      experiment.yaml + run_all.sh + {2D,2_5D,3D}/<model>.sh
│   ├── exp02_data_fraction/    8 arms x MLP head x 25/50/75/100%   experiment.yaml + run_all.sh + 3D/<model>.sh
│   ├── exp03_head_ablation/    6 arms x {MLP, KAN} x 100%          experiment.yaml + run_all.sh + 3D/<model>.sh
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
| `TASK=prognosis` | `diagnosis` | image-only prognosis (`TARGET`, `COHORT=all` or `pe`) |
| `DRY_LIST=1` | off | print the case grid only |

One case directly: `python tools/baselines/run_case.py --model vit_3d --head kan --fraction 50 --fold 2 --gpus 1`.

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

## K-fold and training fractions

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
│   └── preview/  NN_<patient>_<study>_<TP|TN|FP|FN>.{html,png}, _ct.nii.gz + _gradcam.nii.gz
│                 (input grid, sidecar affine; open together in ITK-SNAP / 3D Slicer)
│                 (+ _mil_attention.png for 2D / 2.5D)
├── exp01_baselines/      summary.md  summary.csv  runs.csv  auroc_per_model.png  launcher_logs/
│                         runs/<model>_<head>[_frac<PPP>]/<fold>_seed<S> -> symlink to the shared run
├── exp02_data_fraction/  ...                                  auroc_vs_fraction.png
└── exp03_head_ablation/  ...                                  mlp_vs_kan.png
```

A run is identified by its settings, not by the experiment, so e.g. ResNet-18 3D + MLP at
100% is trained once and reused by all three experiments.

The preview explains the **3 most confident correct (TP/TN alternating) and 3 most confident
wrong (FP/FN alternating) validation cases** (`preview:` in `baselines.yaml`), with the
validation threshold. The Grad-CAM target is each encoder's deepest feature map: conv stage
for CNN/Swin/VMamba/PENet, the last token map reshaped to its 3D grid for ViT / Mamba-MAE,
and for 2D / 2.5D the per-slice maps stacked along z; their montage selects the highest
MIL-attention slices. Cases are
taken from validation (as in the existing CT-FM previews), so test stays untouched.
Official-split and k-fold runs are summarised in separate rows (`split` column).
`python tools/tasks/gradcam_preview.py --run-dir <run>` rebuilds a preview.
