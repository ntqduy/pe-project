# Pipeline

```text
DATA
 └─> FOUNDATION SCREENING
      └─> REPRESENTATION ADAPTATION
           └─> DIAGNOSIS / PROGNOSIS
                └─> ANATOMY ANALYSIS
```

Each stage produces an artifact the next stage consumes. Nothing runs a prerequisite for you:
`python run.py plan <experiment>` tells you what is missing, and preflight refuses to start a
run whose inputs are absent.

## DATA

INSPECT preprocessing, one explicit patient-level split, native labels, EHR/PESI, plus two
**support artifacts**:

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

## FOUNDATION SCREENING

Three public candidate encoders: CT-FM, CT-CLIP, TotalFM.

For an **image-only encoder the screening protocol is: frozen encoder → linear probe.** That
is the only claim the model supports. CT-FM and TotalFM are image-only, so they get probes
and nothing else — there is no "CT-FM zero-shot PE classification" arm, because the model has
no text head to prompt.

CT-CLIP is a joint image/text model, so zero-shot PE prediction is conceivable. It is
registered as `repr.foundation.ct_clip_zero_shot` with status `unavailable`: the text tower,
prompt set, score calibration and evaluation entrypoint do not exist here, and foundation
materialization is not a substitute for them. The contract is written down; nothing is faked.

## REPRESENTATION ADAPTATION

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
python run.py run probe.diag --gpus 0 \
  --set lineage.source_experiment=D_dapt_dino \
  --set lineage.source_checkpoint='${PE_CLOUD_ROOT}/pe-project/outputs/pretraining/dapt/D_dapt_dino/best.ckpt' \
  --set experiment.id=DX_probe_diagnosis_dapt_dino
```

Checkpoint selection uses **validation** performance. Never select a representation, an epoch
or a threshold on the test split.

## DIAGNOSIS

Image-only inference. **Diagnosis never uses EHR or PESI** — a diagnosis model that reads the
clinical record is no longer answering "can this be read off the scan".

Main target: PE +/-. The axes, each isolated by one pair of arms:

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

`prog.global.concat` is deliberately an alias of `prog.image_ehr_pesi` — the same run, not a
duplicate. The full grid of representation × modality × supervision × anatomy × fusion is
**not** built: each listed arm isolates one axis against a fixed reference.

Clinical preprocessing (imputation, normalization) is fit on the training split only, and
prognosis evaluation reports calibration alongside discrimination.

## ANATOMY ANALYSIS

Two different questions, which need two different experiment types.

**Necessity — `anatomy.remove_*`.** Take the *same trained model*
(`diag.anatomy.concat`), freeze every parameter, and score each patient twice: on the original
volume and with one region erased using the shared local-mean replacement policy. Report the
paired probability delta on the same patients. No retraining, no re-segmentation, original
masks reused as stored. `anatomy.remove_random` erases a volume-matched random region and is
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
