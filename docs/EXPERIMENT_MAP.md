# Experiment map

One row per experiment in the active pipeline. Generated from [`configs/experiments.yaml`](../configs/experiments.yaml), which is the source of truth —
update the registry first, then this table.

`anatomy` says whether the image branch is split into heart/PA/lung.
`requires` lists producing experiments and unresolved contracts, not every file; run `python run.py plan <experiment>` for the full chain with READY/MISSING/BLOCKED.

Status: `ready` = config complete; `blocked` = an unresolved data/checkpoint contract (see [BLOCKERS.md](BLOCKERS.md));
`deferred` = outside the active pipeline; `unavailable` = cannot run by design.

Every row runs against either dataset profile (`--set data.profile=test_500_sample|full_inspect`),
against any registered backbone (`--set model.backbone=...`) and — for the arms with an image
encoder — from any encoder initialization (`--set encoder.init_source=pretrained|dapt|c0|silver`).
Those three choices are stamped into the run id, so each combination has its own output
directory. The rows below list each arm's *baseline* configuration.

## DATASET

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `data.dataset.full_inspect` | Which patients and studies does this project actually train and evaluate on? | none | dataset | image, reports, labels | no | none | none | the read-only INSPECT release | ctpa/diagnosis/prognosis/paired_reports/reports manifests, exclusions.csv, integrity.json, split_audit.json, dataset.json, volume cache | ready |
| `data.dataset.test_500_sample` | Does the whole pipeline run end to end before it is run at full cost? | none | dataset | image, reports, labels | no | none | none | the read-only INSPECT release | the same manifest set for 500 patients, sampled patient-level inside the official split | ready |

## DATA

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `data.segmentation`<br>`SEG01` | Which studies have usable anatomy masks? | none | segmentation | image | produces | none | none | `data.dataset.*` (CTPA manifest); contract: segmentation_backend | anatomy masks, mask manifest, QC report, preview images | ready |
| `data.roi`<br>`ROI01` | Which studies have usable ROI artifacts and matched random controls? | none | roi | image | produces | none | none | `data.segmentation` | ROI1-ROI8 masks, ROI manifest, QC report | ready |
| `data.silver.medgemma`<br>`SL00_medgemma_only` | What supervision can a single medical LLM extract from the reports? | none | silver | reports | no | none | none | `data.dataset.*` (reports table); contract: silver_model_weights | accepted/abstained/no_result labels, audit trail, QC summary | blocked |
| `data.silver.rules_falcon`<br>`SL01_rules_falcon` | Does a rule-first cascade extract report supervision more reliably than an LLM alone? | none | silver | reports | no | none | none | `data.dataset.*` (reports table); contract: silver_model_weights | accepted/abstained/no_result labels, audit trail, QC summary | blocked |
| `data.silver.hybrid`<br>`SL02_hybrid` | Does two-model adjudication produce more trustworthy accepted labels? | none | silver | reports | no | none | none | `data.dataset.*` (reports table); contract: silver_model_weights | accepted silver labels used by the multitask arms, audit trail, QC summary | blocked |

## FOUNDATION

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `repr.foundation.ct_fm` | How much PE signal does the public CT-FM representation already carry? | public CT-FM | diagnosis probe | image | no | native PE label | none (fixed linear head) | `data.dataset.*` (diagnosis manifest); contract: backbone_contract | probe validation/test metrics, probe checkpoint | blocked |
| `repr.foundation.ct_clip` | Does the CT-CLIP image tower carry more PE signal than CT-FM or TotalFM? | public CT-CLIP | diagnosis probe | image | no | native PE label | none (fixed linear head) | `data.dataset.*` (diagnosis manifest); contract: backbone_contract | probe validation/test metrics, probe checkpoint | blocked |
| `repr.foundation.totalfm` | Does the TotalFM representation carry more PE signal than CT-FM or CT-CLIP? | public TotalFM | diagnosis probe | image | no | native PE label | none (fixed linear head) | `data.dataset.*` (diagnosis manifest); contract: backbone_contract | probe validation/test metrics, probe checkpoint | blocked |
| `repr.foundation.ct_clip_zero_shot` | Can the joint image/text model predict PE with no PE labels at all? | public CT-CLIP | zero-shot classification | image, text prompts | no | none | none | contract: zero_shot_contract | - | unavailable |

## REPRESENTATION

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `repr.dapt.none`<br>`D00` | What does the public encoder score before any in-domain adaptation? | public | dapt | image | no | none | none | `data.dataset.*` (CTPA manifest); contract: backbone_contract | control encoder checkpoint | blocked |
| `repr.dapt.mae`<br>`D01` | Does masked reconstruction improve the fixed downstream probes? | public + MAE | dapt | image | no | self-supervised | none | `data.dataset.*` (CTPA manifest); contract: backbone_contract | DAPT encoder checkpoint | blocked |
| `repr.dapt.dino`<br>`D02` | Does DINO improve the fixed downstream probes over no adaptation and over MAE? | public + DINO | dapt | image | no | self-supervised | none | `data.dataset.*` (CTPA manifest); contract: backbone_contract | DAPT encoder checkpoint used by the historical alignment lineage | blocked |
| `repr.dapt.simclr`<br>`D03` | Does instance-contrastive adaptation improve the fixed downstream probes? | public + SimCLR | dapt | image | no | self-supervised | none | `data.dataset.*` (CTPA manifest); contract: backbone_contract | DAPT encoder checkpoint | blocked |
| `repr.dapt.anatomy`<br>`D04` | Does anatomy-aware pretraining teach features the fixed probes can detect? | public + anatomy-DAPT | dapt | image | yes | self-supervised | none | `data.dataset.*` (CTPA manifest); `data.segmentation`; contract: backbone_contract | anatomy-aware DAPT encoder checkpoint | blocked |
| `repr.align`<br>`AL01` | Does report alignment improve the representation beyond self-supervised DAPT? | C0 | alignment | image, report embeddings | no | image-report pairing | none | `data.dataset.*` (paired report manifest); `repr.dapt.dino`; contract: report_embedding_columns | C0 checkpoint - the shared starting point for the downstream tasks | blocked |
| `repr.rspect.single_task`<br>`DX06_rsna_single` | Does external supervised transfer produce a better representation than C0 alone? | C0 + external supervision | diagnosis | image | no | external native labels | soft_moe | contract: rspect_manifest; `repr.align`; `data.segmentation` | externally supervised encoder checkpoint - an optional side branch | blocked |
| `repr.rspect.multitask`<br>`DX07_rsna_multi` | Does multitask external supervision transfer better than presence-only? | C0 + external multitask supervision | diagnosis | image | no | external native multitask labels | soft_moe | contract: rspect_manifest; `repr.align`; `data.segmentation` | externally supervised encoder checkpoint - an optional side branch | blocked |
| `repr.silver.hybrid`<br>`SE01` | Does accepted report-derived supervision improve the representation over C0? | C_silver | silver encoder adaptation | image | no | accepted silver multitask | none | `repr.align`; `data.silver.hybrid`; CTPA manifest | C_silver checkpoint - an optional side branch, never a prerequisite | blocked |
| `repr.silver.medgemma` | Does the silver source change what the adapted representation learns? | C_silver (SL00) | silver encoder adaptation | image | no | accepted silver multitask | none | `repr.align`; `data.silver.medgemma`; CTPA manifest | C_silver (SL00) checkpoint | blocked |
| `repr.silver.rules_falcon` | Does the rule-first silver source produce a better adapted representation? | C_silver (SL01) | silver encoder adaptation | image | no | accepted silver multitask | none | `repr.align`; `data.silver.rules_falcon`; CTPA manifest | C_silver (SL01) checkpoint | blocked |

## PROBE

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `probe.diag` | Which representation checkpoint carries the most PE information? | any (`encoder.init_source` / `encoder.checkpoint`) | diagnosis probe | image | no | native PE label | none (fixed linear head) | `data.dataset.*` (diagnosis manifest); `repr.align` | comparable probe metrics per representation checkpoint | blocked |
| `probe.prog` | Which representation checkpoint carries the most outcome-relevant information? | any (`encoder.init_source` / `encoder.checkpoint`) | prognosis probe | image | no | native mortality label | none (fixed small head) | Prognosis manifest with patient-level split; `repr.align` | comparable probe metrics per representation checkpoint | blocked |

## DIAGNOSIS

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `diag.report_only`<br>`DX04_report_only` | How much of the diagnosis score is achievable from the report alone? | none (report embeddings) | diagnosis | report embeddings | no | native multitask | none | Diagnosis manifest with report embedding columns; contract: report_embedding_columns | text-only reference metrics that bound the image results | blocked |
| `diag.global.single` | How well does whole-volume image evidence alone predict native PE labels? | C0 | diagnosis | image | no | native PE label | concat_mlp | `data.dataset.*` (diagnosis manifest); `repr.align` | the reference metrics every anatomy and multitask arm is compared against | blocked |
| `diag.global.native_multitask` | Does reviewed native attribute supervision improve the primary PE head? | C0 | diagnosis | image | no | native multitask | concat_mlp | contract: native_attribute_columns; `repr.align` | multitask diagnosis metrics comparable to diag.global.single | blocked |
| `diag.global.silver_multitask` | Does accepted report-derived supervision improve the primary PE head? | C0 | diagnosis | image | no | native PE plus accepted silver | concat_mlp | `data.dataset.*` (diagnosis manifest); `repr.align`; `data.silver.hybrid` | silver-multitask diagnosis metrics | blocked |
| `diag.anatomy.concat`<br>`DX18_ANATOMY_FULL` | Does pooling the feature map into heart/PA/lung branches beat global-only? | C0 | diagnosis | image | yes | native PE label | concat_mlp | `data.dataset.*` (diagnosis manifest); `repr.align`; `data.segmentation` | anatomy-aware diagnosis checkpoint, the frozen model every counterfactual uses and the teacher every KD student uses | blocked |
| `diag.anatomy.silver.concat` | Does supervising each organ branch with its own silver findings improve PE diagnosis? | C0 | diagnosis | image | yes | native PE plus accepted silver on organ branches | concat_mlp | `data.dataset.*` (diagnosis manifest); `repr.align`; `data.segmentation`; `data.silver.hybrid` | anatomy silver-multitask metrics, concat cell of the fusion comparison | blocked |
| `diag.anatomy.silver.late` | Is combining branch decisions better than combining branch features? | C0 | diagnosis | image | yes | native PE plus accepted silver on organ branches | late_logit | `data.dataset.*` (diagnosis manifest); `repr.align`; `data.segmentation`; `data.silver.hybrid` | late-logit cell of the fusion comparison | blocked |
| `diag.anatomy.silver.moe`<br>`DX17_MULTITASK_SILVER` | Does a learned per-patient router beat fixed concatenation and logit averaging? | C0 | diagnosis | image | yes | native PE plus accepted silver on organ branches | soft_moe | `data.dataset.*` (diagnosis manifest); `repr.align`; `data.segmentation`; `data.silver.hybrid` | Soft-MoE cell of the fusion comparison | blocked |

## PROGNOSIS

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `prog.pesi`<br>`PR26_PESI_ONLY` | How much 30-day mortality signal is already in the established clinical score? | none (tabular) | prognosis | pesi | no | native mortality label | none | `data.dataset.*` (prognosis manifest); PESI columns present in the manifest | the clinical baseline every imaging arm must beat | blocked |
| `prog.ehr`<br>`PR27_CLINICAL_ONLY` | Does the full clinical record beat the PESI score that summarizes part of it? | none (tabular) | prognosis | ehr | no | native mortality label | none | `data.dataset.*` (prognosis manifest); contract: ehr_columns | clinical-only mortality metrics | blocked |
| `prog.image`<br>`PR18_IMAGE_FULL` | Does the CTPA itself carry mortality information beyond the clinical baselines? | C0 | prognosis | image | no | native mortality label | concat_mlp | `data.dataset.*` (prognosis manifest); `repr.align` | image-only mortality metrics | blocked |
| `prog.ehr_pesi` | What is the best purely tabular model? | none (tabular) | prognosis | ehr, pesi | no | native mortality label | concat_mlp | `data.dataset.*` (prognosis manifest); contract: ehr_columns; PESI columns present in the manifest | the tabular bar every imaging arm must clear | blocked |
| `prog.image_pesi` | Does imaging add anything on top of the established severity score? | C0 | prognosis | image, pesi | no | native mortality label | concat_mlp | `data.dataset.*` (prognosis manifest); `repr.align`; PESI columns present in the manifest | image + score mortality metrics | blocked |
| `prog.image_ehr`<br>`PR19_IMAGE_CLINICAL_FULL` | Does imaging add information beyond the full clinical record? | C0 | prognosis | image, ehr | no | native mortality label | concat_mlp | `data.dataset.*` (prognosis manifest); `repr.align`; contract: ehr_columns | image + clinical mortality metrics | blocked |
| `prog.image_ehr_pesi`<br>`PR20_IMAGE_CLINICAL_PESI_FULL` | Does the full multimodal model beat every smaller modality subset? | C0 | prognosis | image, ehr, pesi | no | native mortality label | concat_mlp | `data.dataset.*` (prognosis manifest); `repr.align`; contract: ehr_columns; PESI columns present in the manifest | full multimodal mortality metrics | blocked |
| `prog.global.concat` (alias of `prog.image_ehr_pesi`) | Does concatenation fuse the three modalities better than late-logit or Soft-MoE? | C0 | prognosis | image, ehr, pesi | no | native mortality label | concat_mlp | `prog.image_ehr_pesi` | nothing new - running prog.image_ehr_pesi produces this cell | blocked |
| `prog.global.late` | Is averaging modality decisions better than concatenating modality features? | C0 | prognosis | image, ehr, pesi | no | native mortality label | late_logit | `data.dataset.*` (prognosis manifest); `repr.align`; contract: ehr_columns; PESI columns present in the manifest | late-logit cell of the global fusion comparison | blocked |
| `prog.global.moe` | Does a learned router over modalities beat fixed fusion? | C0 | prognosis | image, ehr, pesi | no | native mortality label | soft_moe | `data.dataset.*` (prognosis manifest); `repr.align`; contract: ehr_columns; PESI columns present in the manifest | Soft-MoE cell of the global fusion comparison | blocked |
| `prog.anatomy.concat`<br>`PR22_ANATOMY_CLINICAL_PESI` | Does anatomy-partitioned image evidence predict mortality better than a global vector? | C0 | prognosis | image, ehr, pesi | yes | native mortality label | concat_mlp | `data.dataset.*` (prognosis manifest); `repr.align`; `data.segmentation`; contract: ehr_columns; PESI columns present in the manifest | anatomy-aware multimodal mortality metrics | blocked |
| `prog.anatomy.late` | With six branches, is averaging decisions better than concatenating features? | C0 | prognosis | image, ehr, pesi | yes | native mortality label | late_logit | `data.dataset.*` (prognosis manifest); `repr.align`; `data.segmentation`; contract: ehr_columns; PESI columns present in the manifest | late-logit cell of the anatomy fusion comparison | blocked |
| `prog.anatomy.moe` | Does a learned router over anatomy and clinical branches beat fixed fusion? | C0 | prognosis | image, ehr, pesi | yes | native mortality label | soft_moe | `data.dataset.*` (prognosis manifest); `repr.align`; `data.segmentation`; contract: ehr_columns; PESI columns present in the manifest | Soft-MoE cell of the anatomy fusion comparison | blocked |

## ANATOMY ANALYSIS

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `anatomy.remove_heart`<br>`CF01_REMOVE_HEART` | Is heart evidence necessary for the trained model? | frozen diag.anatomy.concat | counterfactual inference | image | yes | none (no training) | concat_mlp | `diag.anatomy.concat`; `data.segmentation` | paired original vs removal probabilities, probability deltas | blocked |
| `anatomy.remove_pa`<br>`CF02_REMOVE_PA` | Is pulmonary-artery evidence necessary for the trained model? | frozen diag.anatomy.concat | counterfactual inference | image | yes | none (no training) | concat_mlp | `diag.anatomy.concat`; `data.segmentation` | paired original vs removal probabilities, probability deltas | blocked |
| `anatomy.remove_lung`<br>`CF03_REMOVE_LUNG` | Is lung-parenchyma evidence necessary for the trained model? | frozen diag.anatomy.concat | counterfactual inference | image | yes | none (no training) | concat_mlp | `diag.anatomy.concat`; `data.segmentation` | paired original vs removal probabilities, probability deltas | blocked |
| `anatomy.remove_random`<br>`CF04_REMOVE_RANDOM` | How much does erasing any region of that size move the score? | frozen diag.anatomy.concat | counterfactual inference | image | yes | none (no training) | concat_mlp | `diag.anatomy.concat`; `data.segmentation`; `data.roi` | the control deltas the organ removals must be read against | blocked |
| `anatomy.heart_student`<br>`RS06_HEART_GT` | Is the heart region alone sufficient to predict PE? | C0 | diagnosis (ROI student) | image | yes | native PE label | concat_mlp | `repr.align`; `data.roi`; Diagnosis manifest with patient-level split | heart-only sufficiency metrics | blocked |
| `anatomy.pa_student`<br>`RS07_PA_GT` | Is the pulmonary-artery region alone sufficient to predict PE? | C0 | diagnosis (ROI student) | image | yes | native PE label | concat_mlp | `repr.align`; `data.roi`; Diagnosis manifest with patient-level split | PA-only sufficiency metrics | blocked |
| `anatomy.lung_student`<br>`RS08_LUNG_GT` | Is the lung parenchyma alone sufficient to predict PE? | C0 | diagnosis (ROI student) | image | yes | native PE label | concat_mlp | `repr.align`; `data.roi`; Diagnosis manifest with patient-level split | lung-only sufficiency metrics | blocked |
| `anatomy.random_student`<br>`RS09_RANDOM_GT` | How much of an organ student's score is dataset shortcut rather than anatomy? | C0 | diagnosis (ROI student) | image | yes | native PE label | concat_mlp | `repr.align`; `data.roi`; Diagnosis manifest with patient-level split | the control the organ students must beat | blocked |
| `anatomy.heart_student_kd`<br>`KD01_HEART` | Does the full-volume teacher transfer knowledge the heart ROI labels cannot? | C0 student, frozen full-model teacher | diagnosis (ROI student + KD) | image | yes | native PE label plus teacher logits | concat_mlp | `repr.align`; `diag.anatomy.concat`; `data.roi` | distilled heart-only metrics with GT/KD/total loss components | blocked |
| `anatomy.pa_student_kd`<br>`KD02_PA` | How close to the full model can a PA-only student get with a teacher? | C0 student, frozen full-model teacher | diagnosis (ROI student + KD) | image | yes | native PE label plus teacher logits | concat_mlp | `repr.align`; `diag.anatomy.concat`; `data.roi` | distilled PA-only metrics with GT/KD/total loss components | blocked |
| `anatomy.lung_student_kd`<br>`KD03_LUNG` | Does distillation lift the lung-only student, and by how much? | C0 student, frozen full-model teacher | diagnosis (ROI student + KD) | image | yes | native PE label plus teacher logits | concat_mlp | `repr.align`; `diag.anatomy.concat`; `data.roi` | distilled lung-only metrics with GT/KD/total loss components | blocked |
| `anatomy.random_student_kd`<br>`KD04_RANDOM` | How much can distillation lift a student that sees nothing anatomically relevant? | C0 student, frozen full-model teacher | diagnosis (ROI student + KD) | image | yes | native PE label plus teacher logits | concat_mlp | `repr.align`; `diag.anatomy.concat`; `data.roi` | the control the distilled organ students must beat | blocked |

## DEFERRED

| experiment | scientific_question | representation | task | modalities | anatomy | supervision | fusion | requires | produces | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `deferred.contour`<br>`CT01` | Can the shared representation localize the embolus, not just detect it? | C_diag | contour | image | no | native embolus mask | none | Contour manifest with reviewed embolus mask paths; Source diagnosis checkpoint | contour checkpoint, Dice/NSD/HD95 metrics | deferred |
| `deferred.concept_bottleneck`<br>`PR17_concept_bottleneck` | Can mortality be predicted through named clinical concepts instead of latent features? | C_diag | prognosis | image, ehr | yes | native labels plus concept targets | none | contract: concept_targets | concept predictions plus mortality prediction | deferred |

## Run ids and output directories

`experiment.id` is the output key: it names the run directory under the output root and is
stamped into every checkpoint's lineage. Ids are self-describing, so a path under `outputs/`
says what produced it without opening a config. The third column is the pre-refactor id the
arm replaces, kept so older notes and result folders can still be traced. (This table is
maintained by hand; keep it in step with the configs.)

| semantic name | output directory | replaces |
|---|---|---|
| `data.segmentation` | `outputs/segmentation/SEG_pseudo_anatomy/` | `SEG01` |
| `data.roi` | `outputs/roi/ROI_anatomy_and_controls/` | `ROI01` |
| `data.silver.medgemma` | `outputs/silver_label/SL_medgemma_only/` | `SL00_medgemma_only` |
| `data.silver.rules_falcon` | `outputs/silver_label/SL_rules_falcon/` | `SL01_rules_falcon` |
| `data.silver.hybrid` | `outputs/silver_label/SL_hybrid_adjudicated/` | `SL02_hybrid` |
| `repr.foundation.ct_fm` | `outputs/diagnosis/DX_probe_ct_fm/` | new arm |
| `repr.foundation.ct_clip` | `outputs/diagnosis/DX_probe_ct_clip/` | new arm |
| `repr.foundation.totalfm` | `outputs/diagnosis/DX_probe_totalfm/` | new arm |
| `repr.foundation.ct_clip_zero_shot` | `outputs/pretraining/foundation/F_ct_clip_zero_shot/` (never written: unavailable) | new arm |
| `repr.dapt.none` | `outputs/pretraining/dapt/D_dapt_none/` | `D00` |
| `repr.dapt.mae` | `outputs/pretraining/dapt/D_dapt_mae/` | `D01` |
| `repr.dapt.dino` | `outputs/pretraining/dapt/D_dapt_dino/` | `D02` |
| `repr.dapt.simclr` | `outputs/pretraining/dapt/D_dapt_simclr/` | `D03` |
| `repr.dapt.anatomy` | `outputs/pretraining/dapt/D_dapt_anatomy/` | `D04` |
| `repr.align` | `outputs/pretraining/alignment/AL_image_report_c0/` (C0) | `AL01` |
| `probe.diag` | `outputs/diagnosis/DX_probe_diagnosis/` | new arm |
| `probe.prog` | `outputs/prognosis/PR_probe_prognosis/` | new arm |
| `repr.rspect.single_task` | `outputs/diagnosis/DX_rspect_single_task/` | `DX06_rsna_single` |
| `repr.rspect.multitask` | `outputs/diagnosis/DX_rspect_multitask/` | `DX07_rsna_multi` |
| `repr.silver.hybrid` | `outputs/pretraining/silver/SE_c_silver_hybrid/` (C_silver) | `SE01` |
| `repr.silver.medgemma` | `outputs/pretraining/silver/SE_c_silver_medgemma/` | new arm |
| `repr.silver.rules_falcon` | `outputs/pretraining/silver/SE_c_silver_rules_falcon/` | new arm |
| `diag.report_only` | `outputs/diagnosis/DX_report_only/` | `DX04_report_only` |
| `diag.global.single` | `outputs/diagnosis/DX_global_single/` | new arm |
| `diag.global.native_multitask` | `outputs/diagnosis/DX_global_native_multitask/` | new arm |
| `diag.global.silver_multitask` | `outputs/diagnosis/DX_global_silver_multitask/` | new arm |
| `diag.anatomy.concat` | `outputs/diagnosis/DX_anatomy_concat/` (teacher / frozen reference) | `DX18_ANATOMY_FULL` |
| `diag.anatomy.silver.concat` | `outputs/diagnosis/DX_anatomy_silver_concat/` | new arm |
| `diag.anatomy.silver.late` | `outputs/diagnosis/DX_anatomy_silver_late_logit/` | new arm |
| `diag.anatomy.silver.moe` | `outputs/diagnosis/DX_anatomy_silver_soft_moe/` | `DX17_MULTITASK_SILVER` |
| `prog.pesi` | `outputs/prognosis/PR_pesi_only/` | `PR26_PESI_ONLY` |
| `prog.ehr` | `outputs/prognosis/PR_ehr_only/` | `PR27_CLINICAL_ONLY` |
| `prog.image` | `outputs/prognosis/PR_image_only/` | `PR18_IMAGE_FULL` |
| `prog.ehr_pesi` | `outputs/prognosis/PR_ehr_pesi/` | new arm |
| `prog.image_pesi` | `outputs/prognosis/PR_image_pesi/` | new arm |
| `prog.image_ehr` | `outputs/prognosis/PR_image_ehr/` | `PR19_IMAGE_CLINICAL_FULL` |
| `prog.image_ehr_pesi` (= `prog.global.concat`) | `outputs/prognosis/PR_image_ehr_pesi/` | `PR20_IMAGE_CLINICAL_PESI_FULL` |
| `prog.global.late` | `outputs/prognosis/PR_global_late_logit/` | new arm |
| `prog.global.moe` | `outputs/prognosis/PR_global_soft_moe/` | new arm |
| `prog.anatomy.concat` | `outputs/prognosis/PR_anatomy_concat/` | `PR22_ANATOMY_CLINICAL_PESI` |
| `prog.anatomy.late` | `outputs/prognosis/PR_anatomy_late_logit/` | new arm |
| `prog.anatomy.moe` | `outputs/prognosis/PR_anatomy_soft_moe/` | new arm |
| `anatomy.remove_heart` | `outputs/counterfactual/CF_remove_heart/` | `CF01_REMOVE_HEART` |
| `anatomy.remove_pa` | `outputs/counterfactual/CF_remove_pa/` | `CF02_REMOVE_PA` |
| `anatomy.remove_lung` | `outputs/counterfactual/CF_remove_lung/` | `CF03_REMOVE_LUNG` |
| `anatomy.remove_random` | `outputs/counterfactual/CF_remove_random/` | `CF04_REMOVE_RANDOM` |
| `anatomy.heart_student` | `outputs/roi_students/RS_heart_only_gt/` | `RS06_HEART_GT` |
| `anatomy.pa_student` | `outputs/roi_students/RS_pa_only_gt/` | `RS07_PA_GT` |
| `anatomy.lung_student` | `outputs/roi_students/RS_lung_only_gt/` | `RS08_LUNG_GT` |
| `anatomy.random_student` | `outputs/roi_students/RS_random_only_gt/` | `RS09_RANDOM_GT` |
| `anatomy.heart_student_kd` | `outputs/roi_students/KD_heart_only_distilled/` | `KD01_HEART` |
| `anatomy.pa_student_kd` | `outputs/roi_students/KD_pa_only_distilled/` | `KD02_PA` |
| `anatomy.lung_student_kd` | `outputs/roi_students/KD_lung_only_distilled/` | `KD03_LUNG` |
| `anatomy.random_student_kd` | `outputs/roi_students/KD_random_only_distilled/` | `KD04_RANDOM` |
| `deferred.contour` | `outputs/contour/CT_embolus_contour/` | `CT01` |
| `deferred.concept_bottleneck` | `outputs/prognosis/PR_concept_bottleneck/` | `PR17_concept_bottleneck` |
