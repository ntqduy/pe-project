# `scripts/` — thin wrappers

Every file here is a wrapper around `run.py`. None of them contains scientific logic:
the experiment lives in `configs/experiments.yaml` + `configs/runs/**`, the code lives in
`source/` and `tools/`. If a change belongs in a script, it belongs in a config instead.

```text
scripts/
├── _lib.sh                shared env -> run.py plumbing
├── 0_data_preprocessing/  build a dataset profile from the read-only INSPECT release
├── 1_segmentation/        TotalSegmentator pseudo-anatomy masks, LungMask QC, ROI crops
├── 2_silver_label/        SL00 / SL01 / SL02 report-derived labels
├── 3_shared_encoder/      run these four sub-stages in the numbered order
│   ├── 0_pretrained_eval/ the public backbones, evaluated before any adaptation
│   ├── 1_dapt/            none / MAE / DINO / SimCLR / anatomy-DAPT
│   ├── 2_alignment/       image-report alignment -> C0
│   └── 3_silver_encoder/  silver adaptation -> C_silver
├── 4_diagnosis/           probe, global, anatomy-aware, ROI students, KD
├── 5_prognosis/           PESI / clinical / image / multimodal / anatomy-aware
└── 6_counterfactual/      frozen-model region removal
```

## The four independent axes

A wrapper never hardcodes any of them.

| axis | environment variable | values |
|---|---|---|
| dataset | `DATASET` | `smoke_30`, `test_500_sample`, `full_inspect` |
| method | which script you run | one experiment each |
| encoder weight | `BACKBONE`, `ENCODER_SOURCE` | `ct_fm`/`ct_clip`/`totalfm` × `pretrained`/`dapt`/`c0`/`silver` |
| run scope | `MAX_CASES`, `PATIENT_ID`, `ALLOW_ALL` | scoped stages only (see below) |

Each combination gets its own output directory: the run id is stamped with whatever
deviates from the config's baseline, e.g. `DX_anatomy_concat__enc_dapt`,
`DX_anatomy_concat__ds_test_500_sample__bb_ct_clip__enc_silver`. Two runs that differ only
in encoder initialization therefore never overwrite each other.

## Test → full

```bash
# smoke: thirty patients (ten per official split) through the generation stages
bash scripts/0_data_preprocessing/build_smoke_30.sh

# the whole 500-patient rehearsal cohort
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_test_500_sample.sh

# the full cohort
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_full_inspect.sh
```

`MAX_CASES` / `PATIENT_ID` / `ALLOW_ALL` apply to the **generation** stages -- dataset
build, segmentation, ROI, silver labels, counterfactual inference -- which is where
`run.py` requires an explicit scope so a full run is never accidental.

A **training** stage has no per-case limit, and none was invented for it: its scope *is*
the dataset profile. Rehearse a training arm on `DATASET=test_500_sample`, and check a run
without starting it with `ACTION=preflight` or `ACTION=dry`. `SMOKE=1` additionally pins
`training.epochs=1`.

```bash
DATASET=test_500_sample ACTION=preflight GPUS=0 bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ACTION=dry SMOKE=1 GPUS=0 bash scripts/4_diagnosis/anatomy_full.sh
DATASET=full_inspect GPUS=0,1 bash scripts/4_diagnosis/anatomy_full.sh
```

## Comparing encoder initializations

The point of the diagnosis programme: same model, same data, same split, same
hyperparameters -- only the weights the encoder starts from change.

```bash
DATASET=test_500_sample ENCODER_SOURCE=pretrained bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=dapt       bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=c0         bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=silver     bash scripts/4_diagnosis/anatomy_full.sh
```

`ENCODER_SOURCE` defaults to the canonical checkpoint of that stage. Point it somewhere
else -- a different DAPT method, a different backbone's DAPT run, a different silver
source -- with `ENCODER_CHECKPOINT` and `ENCODER_EXPERIMENT`:

```bash
ENCODER_SOURCE=dapt \
ENCODER_CHECKPOINT="$PE_CLOUD_ROOT/pe-project/outputs/pretraining/dapt/D_dapt_mae/best.ckpt" \
ENCODER_EXPERIMENT=D_dapt_mae \
bash scripts/4_diagnosis/anatomy_full.sh
```

Switching `BACKBONE` without also giving `ENCODER_CHECKPOINT` is refused for the adapted
sources: the default DAPT/C0/C_silver checkpoints belong to one backbone, and quietly
loading CT-FM weights into a TotalFM run would produce an uninterpretable number.

## Anything else

`SET` passes raw overrides straight through, so nothing needs a new script:

```bash
SET="training.epochs=5 seed=7" DATASET=full_inspect bash scripts/4_diagnosis/anatomy_full.sh
```

The four `3_shared_encoder/` sub-directories are numbered in the order they must run:

```bash
GPUS=0 bash scripts/3_shared_encoder/1_dapt/dino.sh
```

`ACTION` selects the `run.py` verb: `show`, `plan`, `preflight`, `dry`, `run` (default).
`GPUS`, `RESUME=1` and `OVERWRITE=1` are forwarded as-is.
