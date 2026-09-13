# `scripts/` — thin wrappers

Every file here is a wrapper around `run.py`. None of them contains scientific logic:
the experiment lives in `configs/experiments.yaml` + `configs/runs/**`, the code lives in
`source/` and `tools/`. If a change belongs in a script, it belongs in a config instead.

```text
scripts/
├── _lib.sh                shared env -> run.py plumbing
├── _matrix_runner.sh      shared diagnosis/prognosis/RSPECT matrix mapping
├── run_all.sh             enumerate / execute the whole experiment matrix
├── run_cv.sh              one config through patient-stratified K-fold instead of hold-out
├── run_ablation_arch.sh   the six architecture ablation variants, in sequence
├── run_ablation_ehr.sh    the four EHR ablation variants, in sequence
├── run_prognosis_organ.sh the four organ-only prognosis students, in sequence
├── ana.sh                 EDA over a built dataset profile (see analysis/README.md)
├── 0_data_preprocessing/  build a dataset profile from the read-only INSPECT release
├── 1_segmentation/        TotalSegmentator pseudo-anatomy masks, LungMask QC, ROI masks
├── 2_silver_label/        report-derived labels; one script per cascade (rule ... rule_falcon_medgemma)
├── 3_shared_encoder/      run these five sub-stages in the numbered order
│   ├── 0_pretrained_eval/ the public backbones, evaluated before any adaptation
│   ├── 1_dapt/            none / MAE / DINO / SimCLR / anatomy-DAPT
│   ├── 2_alignment/       image-report alignment -> C0
│   ├── 3_rspect/          external supervised transfer: multitask or PE-only
│   └── 4_silver_encoder/  silver adaptation after the selected RSPECT encoder
├── 4_diagnosis/           probe.sh / probe_multitask.sh = frozen-encoder instruments
│   ├── multi-task/        exact three-label protocol launcher (global.sh)
│   └── single-task/       pe_positive / pe_acute / pe_subsegmental,
│                          each with global.sh and probe.sh
├── 5_prognosis/           cohort × EHR-profile × task × strategy symlink matrix,
│                        plus flat wrappers incl. heart/pa/lung/random_student.sh
└── 6_counterfactual/      frozen-model region removal
```

`3_shared_encoder/3_silver_encoder/` is retained only as a forwarding compatibility
path; new commands use `4_silver_encoder/`.

## Protocol matrix

The matrix launchers accept a public weight vocabulary without hard-coded checkpoint
paths:

```text
WEIGHT_SOURCE=pretrained | dapt | alignment | rspect_multitask | rspect_single | silver_encoder | custom
WEIGHT_PATH=/path/to/checkpoint     # required only when WEIGHT_SOURCE=custom
```

The canonical path for every named stage lives once in
`configs/components/encoders.yaml#sources`; `WEIGHT_PATH` overrides it while preserving
the source name in lineage. `alignment` and `silver_encoder` map internally to the
legacy `c0` and `silver` source keys, so archived commands remain valid.

```bash
# RSPECT: external dataset is /mnt/RSPECT_dataset (RSPECT_ROOT may override it).
WEIGHT_SOURCE=alignment ACTION=preflight \
  bash scripts/3_shared_encoder/3_rspect/multi-task/train.sh
WEIGHT_SOURCE=dapt ACTION=preflight \
  bash scripts/3_shared_encoder/3_rspect/single-task/pe_positive.sh

# Diagnosis: all three labels together, or one binary label.
DATASET=test_500_sample WEIGHT_SOURCE=rspect_multitask ACTION=preflight \
  bash scripts/4_diagnosis/multi-task/global.sh
DATASET=test_500_sample WEIGHT_SOURCE=silver_encoder ACTION=preflight \
  bash scripts/4_diagnosis/single-task/pe_acute/global.sh
# ... and the frozen-encoder probe, the fixed instrument for comparing WEIGHT_SOURCE.
DATASET=test_500_sample WEIGHT_SOURCE=dapt ACTION=preflight \
  bash scripts/4_diagnosis/single-task/pe_positive/probe.sh

# Prognosis: choose cohort, strict pre-index EHR profile, one of seven outcomes and a strategy.
DATASET=test_500_sample WEIGHT_SOURCE=alignment ACTION=preflight \
  bash scripts/5_prognosis/all_patient/EHR_0_h/1_month_mortality/image_clinical.sh
DATASET=test_500_sample WEIGHT_SOURCE=rspect_single ACTION=preflight \
  bash scripts/5_prognosis/PE_positive/EHR_24_h/12_month_PH/global_soft_moe.sh
```

`EHR_0_h` means `event_time < CTPA time`. `EHR_24_h` is deliberately stricter:
`event_time < CTPA time - 24 hours`; it is not a post-index 24-hour observation window.
Stage 0 writes both feature tables and `clinical/ehr_profiles.json`; prognosis preflight
requires the selected profile's cutoff, strict operator and exact manifest columns to match.

Matrix outputs are nested and collision checked:

```text
outputs/shared_encoder/rspect/<multitask|single_pe>/weight_<source>/
outputs/shared_encoder/silver_encoder/<silver-label-source>/weight_<source>/
outputs/diagnosis/<multi_task|single_task>/<label>/weight_<source>/<strategy>/
outputs/prognosis/<all_comers|pe_positive_only>/<EHR_0_h|EHR_24_h>/<task>/weight_<source>/<strategy>/
```

Those are the paths of the **protocol run**, `DATASET=full_inspect`. A rehearsal on
another profile is a different experiment and gets `weight_<source>__ds_<profile>/`
instead -- the same rule `stamp_experiment_variant()` applies to the non-matrix stages,
where only a deviation from the default profile is stamped. That is what keeps the
canonical checkpoints named in `components/encoders.yaml#sources` resolvable: the
`silver_encoder` a downstream run loads is the full-cohort one, not whichever rehearsal
was last executed.

RSPECT is the exception with no dataset axis at all: it sets `data.root` to the external
`/mnt/RSPECT_dataset`, which takes precedence over `data.profile`, so `DATASET` cannot
change what it trains on and is deliberately absent from its path.

Every directory has its own `checkpoints/`, `logs/`, resolved config and lineage. A repeat
without `RESUME=1`/`OVERWRITE=1` is rejected rather than silently overwriting it.

`STRATEGY` is validated against the experiments that actually implement it -- diagnosis
accepts `global` (both modes) and `probe` (single-task only), prognosis accepts the
thirteen strategies above plus the four ROI-only students (`heart_student`, `pa_student`,
`lung_student`, `random_student`). The students are deliberately absent from
`run_all.sh`'s default `STRATEGIES` list: they answer a sufficiency question on one
endpoint, not a protocol axis to sweep. An unmapped name is rejected rather than accepted, because
`STRATEGY` names the output directory: silently allowing `STRATEGY=anatomy_soft_moe`
would label a directory as Soft-MoE while training the plain global model. Adding a
strategy means adding a registry experiment, not a string in the launcher.

## Running the matrix

`run_all.sh` iterates the axes and invokes the same per-experiment launchers a human
would run by hand. It prints the plan and exits; executing thousands of trainings is
opt-in.

```bash
bash scripts/run_all.sh                                  # list the plan and the count
EXECUTE=1 ACTION=preflight bash scripts/run_all.sh       # check every run, train nothing
EXECUTE=1 GPUS=0 bash scripts/run_all.sh                 # train the matrix

# any axis can be narrowed
STAGES=diagnosis WEIGHT_SOURCES="dapt alignment" \
  EXECUTE=1 ACTION=dry bash scripts/run_all.sh
STAGES=prognosis COHORTS=pe_positive_only EHR_PROFILES=EHR_24_h \
  TASKS=12_month_PH STRATEGIES=image_clinical_pesi bash scripts/run_all.sh
```

Axes: `STAGES`, `WEIGHT_SOURCES`, `COHORTS`, `EHR_PROFILES`, `TASKS`, `STRATEGIES`,
`DX_MODES`, `DX_LABELS`, `DX_STRATEGIES`, `SILVER_LABELS`. `RSPECT_WEIGHT_SOURCES`
defaults to the three stages that precede RSPECT. `KEEP_GOING=1` continues past a
failing run. `clinical_only`, `pesi_only` and `clinical_pesi` read no image, so the
matrix emits them once rather than retraining an identical tabular model per weight
source; `TABULAR_WEIGHT_SOURCE` chooses which directory they land in.

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
