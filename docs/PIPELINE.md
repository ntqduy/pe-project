# Pipeline

```text
DATASET
 └─> DATA (masks, ROIs, silver labels)
      └─> SHARED ENCODER
           └─> DIAGNOSIS / PROGNOSIS
                └─> ANATOMY ANALYSIS
```

Each stage produces an artifact the next stage consumes. Nothing runs a prerequisite for you:
`python run.py plan <experiment>` tells you what is missing, and preflight refuses to start a
run whose inputs are absent.

## Four independent axes

Every experiment below is one point in a four-dimensional space, and the four dimensions are
kept independent on purpose — otherwise no two results are comparable.

| axis | where it is set | values |
|---|---|---|
| **dataset** | `data.profile` | `test_500_sample`, `full_inspect` |
| **method** | which experiment you run | one config each |
| **encoder weight** | `model.backbone` × `encoder.init_source` | `ct_fm`/`ct_clip`/`totalfm` × `pretrained`/`dapt`/`c0`/`silver` |
| **run scope** | CLI | `--patient-id`, `--max-cases`, `--allow-full` |

None of them is a code path. Changing one changes a config value and gives the run its own
output directory: the run id is stamped with whatever deviates from the config's baseline
(`DX_anatomy_concat__enc_dapt`, `DX_anatomy_concat__ds_test_500_sample__bb_ct_clip__enc_silver`).

## DATASET

Two profiles, one implementation.

```text
Raw INSPECT release (READ-ONLY)
        │
        ▼  source/data_preprocessing  -- eligibility, noise removal, CT integrity,
        │                                label adjudication, split audit, manifests
        ├──────────────────────────────┐
        ▼                              ▼
  test_500_sample                 full_inspect
  (clean cohort + 500 patients)   (whole eligible cohort, no sampling)
```

- `data.dataset.full_inspect` — every study that survives the shared eligibility and QC
  contract. The official train/valid/test split is preserved exactly as the release
  assigns it.
- `data.dataset.test_500_sample` — the *same* cohort, then 500 patients drawn inside each
  official split in proportion to that split's size, stratified on the PE and mortality
  labels. Sampling is patient-level and a sampled patient keeps all of their studies, so no
  patient is split across the subset boundary and no patient changes split.

Both profiles inherit `source/dataset/profiles/_common.yaml`, which is what makes them
comparable: filtering, integrity, adjudication, manifest construction and volume
preprocessing are defined once. `source.dataset.assert_shared_preprocessing()` fails if they
ever diverge, and the preprocessing fingerprint is written into each dataset's
`dataset.json`. There is deliberately no pilot-only and no full-only scientific code.

Each build writes, next to the manifests, `exclusions.csv` (every dropped study and the rule
that dropped it), `integrity.json` (corrupted/missing CT), `split_audit.json` (leakage and
official-split preservation) and `dataset.json` (full provenance).

## DATA

Three **support artifacts** built on top of a dataset profile:

- `data.segmentation` — TotalSegmentator pseudo-anatomy masks, with LungMask contributing an
  independent lung Dice cross-check only. These are pseudo-labels from a public model, not
  expert-validated CTPA segmentations.
- `data.roi` — ROI1–ROI8 derived from the stored segmentation run: ROI2 heart, ROI4 PA,
  ROI6 lung parenchyma, ROI8 a volume-matched, body-constrained random control.
- `data.silver.*` — report-derived silver labels (SL00 MedGemma, SL01 rules+Falcon,
  SL02 adjudicated hybrid), each row `accepted | abstained | no_result`.

**Support artifacts are inputs, never results.** Masks and silver labels are independent of
each other: ROI construction reads a finished segmentation run and never re-segments; silver
generation reads only report text and never touches a mask. Only `accepted` silver rows are
ever trained on, and silver is never used as an evaluation label.

## SHARED ENCODER

```text
ORIGINAL PRETRAINED BACKBONE  (ct_fm | ct_clip | totalfm)
        │
        ├── pretrained_eval ──> baseline metrics for the public weights
        │
        └── DAPT (none | MAE | DINO | SimCLR | anatomy)
                 ↓
            image-report alignment
                 ↓
                 C0
                 ↓
            silver adaptation
                 ↓
             C_silver
```

Backbone and DAPT method are independent: any of `ct_fm`, `ct_clip`, `totalfm` crossed with
any DAPT method is a valid run, and each combination gets its own checkpoint directory.
**DAPT loads the original public weights itself** (`components/encoder/from_pretrained.yaml`),
so no foundation-materialization stage has to be run first.

Every checkpoint records its own lineage: backbone, source checkpoint and its SHA-256,
`encoder_init_source`, DAPT method, alignment flag, silver source, the dataset profile it was
adapted on, and the seed. A downstream result therefore always says which weights it came
from.

### Pretrained baseline evaluation

`scripts/3_shared_encoder/pretrained_eval/{ct_fm,ct_clip,totalfm}.sh` measure the public
weights **before** any adaptation, so every adaptation has something to beat.

For an **image-only encoder the screening protocol is: frozen encoder → linear probe.** That
is the only claim the model supports. The public checkpoints have no PE head, so a metric is
impossible without fitting one; what these arms fit is the fixed probe head from
`components/training/probe.yaml`, with `peft.method: frozen`. **No pretrained weight is
updated** — the backbone is not fine-tuned, and the probe contract is identical for every
checkpoint so the numbers stay comparable. CT-FM and TotalFM are image-only, so they get probes
and nothing else — there is no "CT-FM zero-shot PE classification" arm, because the model has
no text head to prompt.

CT-CLIP is a joint image/text model, so zero-shot PE prediction is conceivable. It is
registered as `repr.foundation.ct_clip_zero_shot` with status `unavailable`: the text tower,
prompt set, score calibration and evaluation entrypoint do not exist here, and foundation
materialization is not a substitute for them. The contract is written down; nothing is faked.

### Adaptation arms

Main line: public encoder → DAPT → image-report alignment → **C0**.

- `repr.dapt.none` is the explicit no-adaptation control; `mae`, `dino`, `simclr` and
  `anatomy` are the adaptation methods (only `anatomy` consumes masks).
- `repr.align` produces **C0**, the shared starting point for diagnosis, prognosis and the
  ROI students.

Two **side branches** — neither is a prerequisite for any downstream task:

- `repr.rspect.*` — supervised transfer from the external RSPECT/RSNA-STR cohort,
  single-task versus multitask.
- `repr.silver.*` — adapt C0 with accepted silver labels, producing **C_silver**
  (one arm per silver source). C0 remains intact and is never overwritten.

### Choosing a representation: the fixed probes

Representations are compared with two fixed probes, not by pretraining loss:

- `probe.diag` — frozen encoder → global pool → fixed linear head → PE +/-
- `probe.prog` — frozen encoder → global pool → fixed small head → 30-day mortality

The probe contract lives in `configs/components/training/probe.yaml`. The patient split,
preprocessing, head architecture, optimizer, epochs, batch size, seed and metrics are fixed
for every probed checkpoint; the only things that may differ are the checkpoint under test and
the experiment id. Break that and the comparison is meaningless.

The same probes apply to the public FM, each DAPT arm, C0, an RSPECT-transferred encoder and
C_silver:

```bash
python run.py run probe.diag --gpus 0 --set encoder.init_source=dapt
python run.py run probe.diag --gpus 0 --set encoder.init_source=pretrained
python run.py run probe.diag --gpus 0 --set encoder.init_source=silver
```

`encoder.init_source` resolves the checkpoint *and* stamps the run id, so the three commands
above cannot overwrite each other. A non-canonical checkpoint (another DAPT method, another
backbone's run) is selected explicitly:

```bash
python run.py run probe.diag --gpus 0 --set encoder.init_source=dapt \
  --set encoder.checkpoint='${PE_CLOUD_ROOT}/pe-project/outputs/pretraining/dapt/D_dapt_mae/best.ckpt' \
  --set encoder.source_experiment=D_dapt_mae
```

Checkpoint selection uses **validation** performance. Never select a representation, an epoch
or a threshold on the test split.

## DIAGNOSIS

Image-only inference. **Diagnosis never uses EHR or PESI** — a diagnosis model that reads the
clinical record is no longer answering "can this be read off the scan".

### Which pretrained/adapted weights are best for diagnosis?

This is the comparison the shared-encoder stage exists to enable, so it is run with **one**
model, not four:

```text
same diagnosis architecture   same dataset profile   same split   same hyperparameters
                                     │
                    only the encoder initialization changes
                                     │
      pretrained  ──  DAPT  ──  C0  ──  C_silver
```

```bash
DATASET=test_500_sample ENCODER_SOURCE=pretrained bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=dapt       bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=c0         bash scripts/4_diagnosis/anatomy_full.sh
DATASET=test_500_sample ENCODER_SOURCE=silver     bash scripts/4_diagnosis/anatomy_full.sh
```

There is deliberately no `diagnosis_pretrained_model.py` / `diagnosis_dapt_model.py` /
`diagnosis_silver_model.py`. Four model files would mean four architectures, and any
difference between the results could then be attributed to the code rather than to the
representation. The checkpoint source is a config variable; the model is the same object.

Every run records `lineage.encoder_init_source` and the source checkpoint's SHA-256, so a
result always states which weights it started from.

### The scientific axes

Main target: PE +/-. Each axis is isolated by one pair of arms:

| axis | arms |
|---|---|
| single-task vs multitask | `diag.global.single` vs `diag.global.native_multitask` / `diag.global.silver_multitask` |
| native vs accepted-silver supervision | `diag.global.native_multitask` vs `diag.global.silver_multitask` |
| global vs anatomy-aware | `diag.global.single` vs `diag.anatomy.concat` |
| fusion type | `diag.anatomy.silver.concat` vs `.late` vs `.moe` |
| text ceiling | `diag.report_only` (never sees the volume) |

In the anatomy-aware arms, per-organ accepted silver findings supervise the branch they belong
to: RV/septal/pericardial findings on the heart branch, acuity and clot location on the PA
branch, effusion and chronic lung disease on the lung branch.

The heart/PA/lung vectors are **mask-pooled views of one full-volume feature map**, not
organ-only inputs. Only the ROI students below actually restrict what the model can see.

## PROGNOSIS

Primary cohort: **confirmed acute PE**. Target: 30-day mortality. One shared cohort contract
(`configs/components/task/prognosis_primary_cohort.yaml`) so every arm sees the same patients.

Modality ablation, seven arms, fixed fusion:

```text
prog.pesi        prog.ehr        prog.image
prog.ehr_pesi    prog.image_pesi prog.image_ehr
prog.image_ehr_pesi
```

Then, with all three modalities fixed, two further comparisons:

- global image vs anatomy-aware image: `prog.image_ehr_pesi` vs `prog.anatomy.concat`
- fusion type: concat vs late-logit vs Soft-MoE, in both the `prog.global.*` and
  `prog.anatomy.*` families

Every prognosis arm with an image encoder takes the same `encoder.init_source` variable as
diagnosis (`pretrained`, `dapt`, `c0`, `silver`), so the representation comparison can be
repeated for the outcome task without touching model code. The tabular-only arms
(`prog.pesi`, `prog.ehr`, `prog.ehr_pesi`) have no encoder and therefore no such variable.

`prog.global.concat` is deliberately an alias of `prog.image_ehr_pesi` — the same run, not a
duplicate. The full grid of representation × modality × supervision × anatomy × fusion is
**not** built: each listed arm isolates one axis against a fixed reference.

Clinical preprocessing (imputation, normalization) is fit on the training split only, and
prognosis evaluation reports calibration alongside discrimination.

## ANATOMY ANALYSIS

Two different questions, which need two different experiment types.

**Necessity — `anatomy.remove_*`.** Take the *same trained* diagnosis or prognosis model
(by default `diag.anatomy.concat`), freeze every parameter, and score each patient twice: on the original
volume and with one region erased using the shared local-mean replacement policy. Report the
paired probability delta on the same patients. No retraining, no re-segmentation, original
masks reused as stored. Counterfactual arms therefore take no `encoder.init_source`: they
score the checkpoint a diagnosis/prognosis run already produced, whatever it was initialized
from. `anatomy.remove_random` erases a volume-matched random region and is
what the organ deltas must be read against.

**Sufficiency — `anatomy.*_student`.** Train a fresh student from C0 that only ever sees one
region, and see how far it gets. Four regions (heart, PA, lung, random control) × two
supervision settings:

- GT-only: `anatomy.pa_student`
- GT + frozen-teacher KD: `anatomy.pa_student_kd`

The pair isolates exactly what the full-volume teacher adds beyond the labels. The teacher
stays in eval mode and never enters the student optimizer.

Necessity and sufficiency are not the same claim: a region can be unnecessary (the model
copes without it) while still being sufficient (it alone would do), and vice versa. Reporting
one and implying the other is the mistake this design exists to prevent.

## Deferred

- `deferred.contour` — clot localization. Needs a reviewed embolus-mask manifest; a different
  question from the necessity/sufficiency programme.
- `deferred.concept_bottleneck` — kept disabled. A concept bottleneck is only honest when
  every concept has a real reviewed label; enabling proxy concepts would produce confident,
  meaningless explanations. Embolic burden and vascular pruning have no defensible target in
  this dataset and are not instantiated.

Both remain implemented and archived-but-runnable, so reviving them is a data question, not a
code question.
