# Experiment map

One row per registered experiment, generated from
[`configs/experiments.yaml`](../configs/experiments.yaml) — that file is the source of
truth; this table is the readable index. Regenerate it with
`python tools/build_experiment_map.py` after adding or renaming an experiment.

`name` is what you pass to `run.py` and what a `scripts/` wrapper resolves.
`legacy_id` is the pre-refactor id an arm replaces, kept so older notes stay traceable.

60 experiments.

## DATASET

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `data.dataset.smoke_30` | `configs/runs/00_data/dataset/smoke_30.yaml` | — | ready | Build a deterministic 30-patient all-split technical smoke cohort |
| `data.dataset.full_inspect` | `configs/runs/00_data/dataset/full_inspect.yaml` | — | ready | Build the whole eligible INSPECT cohort - filtering, QC, split audit, manifests |
| `data.dataset.test_500_sample` | `configs/runs/00_data/dataset/test_500_sample.yaml` | — | ready | The same cohort reduced to 500 patients, sampled patient-level inside the official split |

## DATA

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `data.segmentation` | `configs/runs/00_data/segmentation.yaml` | SEG01 | ready | TotalSegmentator pseudo-anatomy masks with LungMask lung Dice QC |
| `data.roi` | `configs/runs/00_data/roi.yaml` | ROI01 | ready | ROI1-ROI8 crops and volume-matched random controls from a finished segmentation run |
| `data.silver.medgemma` | `configs/runs/00_data/silver/medgemma.yaml` | SL00_medgemma_only | blocked | SL00 report-derived silver labels from MedGemma alone |
| `data.silver.rules_falcon` | `configs/runs/00_data/silver/rules_falcon.yaml` | SL01_rules_falcon | blocked | SL01 deterministic rules plus Falcon extraction |
| `data.silver.hybrid` | `configs/runs/00_data/silver/hybrid.yaml` | SL02_hybrid | blocked | SL02 rules plus Falcon plus MedGemma with agreement adjudication |

## FOUNDATION

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `repr.foundation.ct_fm` | `configs/runs/01_foundation/ct_fm_probe.yaml` | — | blocked | Frozen public CT-FM encoder under the fixed PE probe |
| `repr.foundation.ct_clip` | `configs/runs/01_foundation/ct_clip_probe.yaml` | — | blocked | Frozen public CT-CLIP image tower under the fixed PE probe |
| `repr.foundation.totalfm` | `configs/runs/01_foundation/totalfm_probe.yaml` | — | blocked | Frozen public TotalFM encoder under the fixed PE probe |
| `repr.foundation.ct_clip_zero_shot` | `configs/runs/01_foundation/ct_clip_zero_shot.yaml` | — | unavailable | CT-CLIP zero-shot PE contract, documented but not implemented |

## REPRESENTATION

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `repr.dapt.none` | `configs/runs/02_representation/dapt/none.yaml` | D00 | blocked | No-adaptation control for the DAPT comparison |
| `repr.dapt.mae` | `configs/runs/02_representation/dapt/mae.yaml` | D01 | blocked | Masked-autoencoder domain-adaptive pretraining on INSPECT CTPA |
| `repr.dapt.dino` | `configs/runs/02_representation/dapt/dino.yaml` | D02 | blocked | DINO self-distillation domain-adaptive pretraining on INSPECT CTPA |
| `repr.dapt.simclr` | `configs/runs/02_representation/dapt/simclr.yaml` | D03 | blocked | Two-view contrastive domain-adaptive pretraining |
| `repr.dapt.anatomy` | `configs/runs/02_representation/dapt/anatomy.yaml` | D04 | blocked | Anatomy-aware DAPT contrasting whole-volume against PA-only views |
| `repr.align` | `configs/runs/02_representation/image_report_alignment.yaml` | AL01 | blocked | Symmetric InfoNCE image-report alignment producing C0 |
| `repr.rspect.single_task` | `configs/runs/02_representation/rspect/single_task.yaml` | DX06_rsna_single | blocked | External RSPECT/RSNA-STR supervised transfer, PE presence only |
| `repr.rspect.multitask` | `configs/runs/02_representation/rspect/multitask.yaml` | DX07_rsna_multi | blocked | External RSPECT/RSNA-STR supervised transfer, PE presence plus attributes |
| `repr.silver.hybrid` | `configs/runs/02_representation/silver_adaptation/hybrid.yaml` | SE01 | blocked | Adapt C0 with accepted SL02 silver labels, producing C_silver |
| `repr.silver.medgemma` | `configs/runs/02_representation/silver_adaptation/medgemma.yaml` | — | blocked | Adapt C0 with accepted SL00 silver labels |
| `repr.silver.rules_falcon` | `configs/runs/02_representation/silver_adaptation/rules_falcon.yaml` | — | blocked | Adapt C0 with accepted SL01 silver labels |

## PROBE

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `probe.diag` | `configs/runs/02_representation/probe/diagnosis.yaml` | — | blocked | Fixed frozen-encoder PE diagnosis probe |
| `probe.prog` | `configs/runs/02_representation/probe/prognosis.yaml` | — | blocked | Fixed frozen-encoder 30-day mortality probe |

## DIAGNOSIS

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `diag.report_only` | `configs/runs/03_diagnosis/baseline/report_only.yaml` | DX04_report_only | blocked | Text-only comparison baseline that never sees the CTPA volume |
| `diag.global.single` | `configs/runs/03_diagnosis/baseline/global_single.yaml` | — | blocked | Global full-CTPA image features with a single PE +/- head |
| `diag.global.native_multitask` | `configs/runs/03_diagnosis/baseline/global_native_multitask.yaml` | — | blocked | Global image features with native multitask heads |
| `diag.global.silver_multitask` | `configs/runs/03_diagnosis/baseline/global_silver_multitask.yaml` | — | blocked | Global image features with accepted-silver auxiliary heads |
| `diag.anatomy.concat` | `configs/runs/03_diagnosis/anatomy/single_concat.yaml` | DX18_ANATOMY_FULL | blocked | Anatomy-aware single-task PE diagnosis, the reference model |
| `diag.anatomy.silver.concat` | `configs/runs/03_diagnosis/anatomy/silver_concat.yaml` | — | blocked | Anatomy branches with accepted-silver organ heads, concatenation fusion |
| `diag.anatomy.silver.late` | `configs/runs/03_diagnosis/anatomy/silver_late_logit.yaml` | — | blocked | Same arm with late-logit fusion over branch decisions |
| `diag.anatomy.silver.moe` | `configs/runs/03_diagnosis/anatomy/silver_soft_moe.yaml` | DX17_MULTITASK_SILVER | blocked | Same arm with Soft-MoE router fusion |

## PROGNOSIS

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `prog.pesi` | `configs/runs/04_prognosis/modality/pesi.yaml` | PR26_PESI_ONLY | blocked | PESI/sPESI severity score only |
| `prog.ehr` | `configs/runs/04_prognosis/modality/ehr.yaml` | PR27_CLINICAL_ONLY | blocked | Raw clinical values plus missingness indicators only |
| `prog.image` | `configs/runs/04_prognosis/modality/image.yaml` | PR18_IMAGE_FULL | blocked | Global full-CTPA image only |
| `prog.ehr_pesi` | `configs/runs/04_prognosis/modality/ehr_pesi.yaml` | — | blocked | Clinical record plus PESI, no imaging |
| `prog.image_pesi` | `configs/runs/04_prognosis/modality/image_pesi.yaml` | — | blocked | Global image plus PESI |
| `prog.image_ehr` | `configs/runs/04_prognosis/modality/image_ehr.yaml` | PR19_IMAGE_CLINICAL_FULL | blocked | Global image plus the raw clinical record |
| `prog.image_ehr_pesi` | `configs/runs/04_prognosis/modality/image_ehr_pesi.yaml` | PR20_IMAGE_CLINICAL_PESI_FULL | blocked | All three modalities, global image, concatenation fusion |
| `prog.global.concat` | `configs/runs/04_prognosis/modality/image_ehr_pesi.yaml` | — | blocked | Concat cell of the global fusion comparison - the same run as prog.image_ehr_pesi |
| `prog.global.late` | `configs/runs/04_prognosis/global/late_logit.yaml` | — | blocked | Global image plus EHR plus PESI with late-logit fusion |
| `prog.global.moe` | `configs/runs/04_prognosis/global/soft_moe.yaml` | — | blocked | Global image plus EHR plus PESI with Soft-MoE fusion |
| `prog.anatomy.concat` | `configs/runs/04_prognosis/anatomy/concat.yaml` | PR22_ANATOMY_CLINICAL_PESI | blocked | Anatomy-aware image plus EHR plus PESI, concatenation fusion |
| `prog.anatomy.late` | `configs/runs/04_prognosis/anatomy/late_logit.yaml` | — | blocked | Anatomy-aware multimodal prognosis with late-logit fusion |
| `prog.anatomy.moe` | `configs/runs/04_prognosis/anatomy/soft_moe.yaml` | — | blocked | Anatomy-aware multimodal prognosis with Soft-MoE fusion |

## ANATOMY ANALYSIS

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `anatomy.remove_heart` | `configs/runs/05_anatomy_analysis/counterfactual/remove_heart.yaml` | CF01_REMOVE_HEART | blocked | Frozen-model heart removal (necessity) |
| `anatomy.remove_pa` | `configs/runs/05_anatomy_analysis/counterfactual/remove_pa.yaml` | CF02_REMOVE_PA | blocked | Frozen-model pulmonary-artery removal (necessity) |
| `anatomy.remove_lung` | `configs/runs/05_anatomy_analysis/counterfactual/remove_lung.yaml` | CF03_REMOVE_LUNG | blocked | Frozen-model lung-parenchyma removal (necessity) |
| `anatomy.remove_random` | `configs/runs/05_anatomy_analysis/counterfactual/remove_random.yaml` | CF04_REMOVE_RANDOM | blocked | Frozen-model volume-matched random removal (necessity control) |
| `anatomy.heart_student` | `configs/runs/05_anatomy_analysis/students/gt/heart.yaml` | RS06_HEART_GT | blocked | Heart-only ROI student trained from C0, ground truth only |
| `anatomy.pa_student` | `configs/runs/05_anatomy_analysis/students/gt/pa.yaml` | RS07_PA_GT | blocked | PA-only ROI student trained from C0, ground truth only |
| `anatomy.lung_student` | `configs/runs/05_anatomy_analysis/students/gt/lung.yaml` | RS08_LUNG_GT | blocked | Lung-only ROI student trained from C0, ground truth only |
| `anatomy.random_student` | `configs/runs/05_anatomy_analysis/students/gt/random.yaml` | RS09_RANDOM_GT | blocked | Volume-matched random-region student (sufficiency control) |
| `anatomy.heart_student_kd` | `configs/runs/05_anatomy_analysis/students/kd/heart.yaml` | KD01_HEART | blocked | Heart-only student with frozen-teacher distillation |
| `anatomy.pa_student_kd` | `configs/runs/05_anatomy_analysis/students/kd/pa.yaml` | KD02_PA | blocked | PA-only student with frozen-teacher distillation |
| `anatomy.lung_student_kd` | `configs/runs/05_anatomy_analysis/students/kd/lung.yaml` | KD03_LUNG | blocked | Lung-only student with frozen-teacher distillation |
| `anatomy.random_student_kd` | `configs/runs/05_anatomy_analysis/students/kd/random.yaml` | KD04_RANDOM | blocked | Random-region student with frozen-teacher distillation (control) |

## DEFERRED

| name | config | legacy id | status | what it is |
| --- | --- | --- | --- | --- |
| `deferred.contour` | `configs/runs/90_deferred/contour/train.yaml` | CT01 | deferred | Clot contour / segmentation refinement |
| `deferred.concept_bottleneck` | `configs/runs/90_deferred/concept_bottleneck/prognosis.yaml` | PR17_concept_bottleneck | deferred | Interpretable concept-bottleneck prognosis, disabled by default |

