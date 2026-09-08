# Blockers — the contracts that are deliberately empty

Every entry below is a data or checkpoint contract that **must be filled in from a real,
inspected artifact**. They are left empty on purpose so `run.py preflight` fails loudly
instead of a guessed value silently producing a number nobody can defend.

`configs/experiments.yaml` is the machine-readable source: each experiment lists the
blocker id it waits on, and `project_blockers` at the top of that file carries the two
project-wide ones. This page explains what filling one in actually means.

| id | where | blocks |
|---|---|---|
| `backbone_contract` | `configs/components/backbone/registry.yaml` | every image experiment |
| `segmentation_backend` | `third_party/`, `configs/runs/00_data/segmentation.yaml` | `data.segmentation`, therefore `data.roi` and every anatomy-aware arm |
| `silver_model_weights` | `third_party/weights/`, `configs/components/silver/*.yaml` | `data.silver.*`, therefore every silver-supervised arm |
| `report_embedding_columns` | `configs/components/alignment/image_report.yaml` | `repr.align`, `diag.report_only` |
| `ehr_columns` | `configs/components/task/prognosis_primary_cohort.yaml` | every prognosis arm with `ehr` |
| `pesi_clinical_approval` | `configs/clinical/pesi_spesi_mapping.yaml` | every prognosis arm with `pesi` |
| `native_attribute_columns` | `configs/runs/03_diagnosis/baseline/global_native_multitask.yaml` | `diag.global.native_multitask` |
| `rspect_manifest` | `configs/runs/02_representation/rspect/*.yaml` | the external RSPECT transfer side branch |
| `concept_targets` | `configs/runs/90_deferred/concept_bottleneck/prognosis.yaml` | deferred, disabled on purpose |
| `zero_shot_contract` | `configs/runs/01_foundation/ct_clip_zero_shot.yaml` | unavailable by design, documented not implemented |

---

## 1. `backbone_contract`

`factory`, `output_adapter` and `feature_dim` are empty (`feature_dim: 0`) for **all three**
public encoders in `configs/components/backbone/registry.yaml`. They must be read off the
checkpoint you actually staged:

* `factory` — the import path that constructs the encoder module.
* `output_adapter` — how that module's output becomes a `[B, C, D, H, W]` feature map.
* `feature_dim` — `C` of that feature map. `source/components/anatomy.py` compares it with
  the observed channel count and raises if they disagree, so a wrong value fails fast.

Never copy a value between backbones, and never infer one from a repository name. The
`third_party/repos/{CT-FM,CT-CLIP,TotalFM}` clones are also still empty — see
[`third_party/versions.yaml`](../third_party/versions.yaml) for the pinned commits.

## 2. `segmentation_backend`

`third_party/repos/TotalSegmentator` and `third_party/repos/lungmask` are empty clones and
the offline task weights under `third_party/weights/segmentation/` are absent. Without them
`data.segmentation` cannot run, so `data.roi` cannot run, so no anatomy-aware arm has masks.

This is the longest dependency chain in the repository:

```text
data.dataset.<profile> -> data.segmentation -> data.roi -> diag.anatomy.* / prog.anatomy.*
                                                        -> anatomy.remove_* / anatomy.*_student*
```

## 3. `silver_model_weights`

MedGemma and Falcon ids and revisions are recorded, and Falcon weights are staged under
`third_party/weights/falcon-7b/`. Confirm both providers load before running `data.silver.*`;
`SL02` loads **both** models on every rank.

## 4. `report_embedding_columns`

`alignment.report_embedding_columns` is `[]`. It must name real numeric columns present in
`manifests/paired_reports.csv`, produced by whichever report encoder you commit to.
`repr.align` (C0) and the `diag.report_only` text baseline both fail preflight until then.

## 5. `ehr_columns`

`data.ehr_columns` is `[]` and `task.ehr_input_dim` is 32, and preflight fails unless
`len(ehr_columns) == ehr_input_dim`. The candidate columns are the ones stage 0 writes to
`clinical/ehr_features.csv` and merges into the manifests. Choosing 32 of them is a
scientific decision; `data.ehr_columns_full` / `data.ehr_columns_common` record the full set
and the high-availability subset, and preflight checks that common is a subset of full.

## 6. `pesi_clinical_approval`

Stage 0 always writes `clinical/pesi_status.json` and `clinical/pesi_mapping_audit.json`.
It writes `clinical/pesi_features.csv` **only** when both of these hold:

1. `configs/clinical/pesi_spesi_mapping.yaml` has `clinical_approval.status: approved` and
   all eleven components at `status: approved`; and
2. `pesi.components_table` in the profile points at an approved study-level CSV with a
   `study_id` column plus each component's `input_column`.

Today the mapping is `pending`: there are candidates for birth/sex, heart rate, respiratory
rate, temperature and oxygen saturation, but no verified systolic-BP source, no approved
cancer / heart-failure / chronic-lung / altered-mental-status phenotype, and no unit or
pre-CTPA window review. So a default build ends normally with
`clinical/pesi_status.json: blocked` and no score is fabricated.

## 7. `native_attribute_columns`

`diag.global.native_multitask` trains on thirteen native attribute columns beyond
`pe_present`. Each column name and its class encoding must be verified against the built
manifest before the arm is meaningful. Unlabelled rows are masked out of the loss; they are
never turned into zeros.

## 8. `rspect_manifest`, `concept_targets`, `zero_shot_contract`

* **RSPECT** — the external transfer branch needs its own manifest with the project column
  schema, including the `*_mask_path` columns its configs name. Nothing downstream depends
  on it.
* **Concept bottleneck** — deferred and disabled. Every concept must point at a real
  reviewed column before `concept_bottleneck.enabled` may become `true`.
* **CT-CLIP zero-shot** — documented as `unavailable` on purpose. The public checkpoints
  carry no PE head, so the `pretrained_eval` arms fit a fixed linear probe; that is a probe,
  not zero-shot classification, and must never be reported as one.

---

## What is *not* a blocker

The dataset stage. `data.dataset.smoke_30`, `data.dataset.test_500_sample` and
`data.dataset.full_inspect` are `ready` and need only `numpy`, `nibabel`, `pyarrow` and the
read-only release. Build one first — every other stage reads its manifests.
