# Blockers

Every unresolved real-data or checkpoint contract in one place. These are deliberate: the
configs refuse to guess a column name, a feature dimension or a checkpoint factory, so
preflight fails loudly instead of producing a number nobody can trust.

Nothing here is a code bug. Each item needs a decision or an artifact from outside the repo.

`python run.py plan <experiment>` marks a requirement `BLOCKED` when it maps to one of these,
and `MISSING` when the artifact simply has not been produced yet.

## 1. Public encoder contracts — blocks every image experiment

**Where:** `configs/components/backbone/registry.yaml` (one entry per backbone)
**State:** `factory: ""`, `output_adapter: ""`, `feature_dim: 0` for all three
**Also:** `third_party/repos/{CT-FM,CT-CLIP,TotalFM}` are empty directories and
`third_party/weights/foundation/` does not exist. Neither the code nor the weights are here.

Each registry entry needs, from the actual inspected checkpoint:

- `factory` — import path that builds the model
- `output_adapter` — import path mapping upstream output to `ImageFeatures`
  (5-D `feature_map`, 2-D `global_embedding`)
- `feature_dim` — the real channel count of that feature map
- the weights staged locally under `third_party/weights/foundation/<backbone>/`
- SHA-256 and exact commit/revision recorded in `third_party/versions.yaml`

Do not copy values between backbones and do not infer them from repository names. The
registry keeps the three contracts side by side precisely so a wrong value is visible. TotalFM is
organ-patch based upstream, so its whole-volume pooling behaviour must be verified before its
probe means anything.

**Unblocks:** every foundation probe, every DAPT arm, alignment, both fixed probes, all
diagnosis and prognosis image arms, and everything downstream of them.

## 2. INSPECT manifests and the patient-level split — RESOLVED as a contract, not yet built

**Where:** `${PE_CLOUD_ROOT}/data/derived/datasets/<profile>/manifests/`
**Produced by:** `data.dataset.full_inspect` / `data.dataset.test_500_sample`

This is no longer a manual step. `source/data_preprocessing` reads the read-only release,
applies the shared eligibility/QC contract and writes `ctpa.csv`, `diagnosis.csv`,
`prognosis.csv`, `paired_reports.csv` and `reports.csv` with `patient_id`, `study_id`,
`split`, `image_path`.

**The split is not created — it is preserved.** INSPECT's official `train/valid/test`
assignment is carried through unchanged (`valid` is renamed to `validation` to match the
project schema), and `split_audit.json` fails the build if any patient crosses a split or
any study's split differs from the release.

```bash
DATASET=full_inspect ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_full_inspect.sh
```

`tools/data/create_split.py` remains for cohorts that arrive with no split of their own
(the external RSPECT cohort, §8). It is never invoked implicitly, and the test split is
never used to select checkpoints or thresholds.

**Still outstanding:** the datasets have not been built yet, so every downstream `plan`
reports MISSING until one of the two profiles has been run.

## 3. Native auxiliary label columns

**Where:** `configs/runs/03_diagnosis/baseline/global_native_multitask.yaml`,
`configs/runs/02_representation/rspect/multitask.yaml`

Each native attribute column (`acuity`, `central`, `lobar`, `segmental`, `subsegmental`,
`saddle`, `rv_enlargement`, `rv_lv_abnormal`, `septal_bowing`, `reflux`, `pleural_effusion`,
`pericardial_effusion`, `chronic_lung_disease`) must be confirmed present in the real manifest
with its class encoding checked — `acuity` in particular is declared as 4-class. Rows without
a label are masked out of the loss; a missing target is never synthesized.

## 4. Report embedding columns

**Where:** `configs/components/alignment/image_report.yaml` (`report_embedding_columns: []`)

Blocks `repr.align` (and therefore C0 and everything after it) and `diag.report_only`. The
columns must be precomputed report embeddings present in the manifest; no text encoder runs
inside these stages.

## 5. Prognosis cohort, EHR and PESI contract

**Where:** `configs/components/task/prognosis_primary_cohort.yaml`

- `ehr_columns` — the EHR variables fed to the clinical encoder. Preflight fails unless the
  count equals `task.ehr_input_dim` (currently 32).
- `ehr_columns_full` — every configured EHR variable.
- `ehr_columns_common` — the dataset-defined high-availability subset; preflight fails if it
  is not a subset of `ehr_columns_full`.
- `pesi_columns` — `[pesi, spesi]` is the historical contract; confirm both exist.
- The `confirmed_acute_pe` cohort manifest itself, with 30-day mortality.
- An acquisition-date column, only if temporal hold-out is ever wanted; leave
  `evaluation.split_strategy: patient_holdout` until such a column is verified.

Fill these once here and all thirteen prognosis arms pick them up.

## 6. Silver-label model weights — mostly resolved

**Where:** `third_party/weights/falcon-7b/`, `third_party/weights/medgemma/`

Both are staged as complete local Hugging Face directories and the configs now point at
them. Verified from the files on disk, not inferred:

| model | id | revision | architecture |
|---|---|---|---|
| Falcon | `tiiuae/falcon-7b` | `8782b5c5d8c9290412416618f36a133653e85285` | `FalconForCausalLM` |
| MedGemma | `google/medgemma-1.5-4b-it` | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` | `Gemma3ForConditionalGeneration` |

MedGemma 1.5 is an image-text-to-text checkpoint, so `auto_model_class` is
`AutoModelForImageTextToText`; `AutoModelForCausalLM` does not resolve
`Gemma3ForConditionalGeneration`. Report mining is text-only, so `AutoTokenizer` stays.

**Still outstanding:** per-shard SHA-256 is unrecorded in `third_party/versions.yaml`
(`checksum_sha256: null`), and neither model has been loaded once to confirm the JSON
response contract. SL02 loads both on every GPU rank — check VRAM before a full run.

## 7. Segmentation backend

**Where:** `configs/runs/00_data/segmentation.yaml`

**Only TotalSegmentator + LungMask are supported.** There is no `SEG02`, no segmentation
fine-tuning and no supervised segmentation training arm, because there are no expert masks
(see §11). TotalSegmentator produces the canonical pseudo-anatomy masks; LungMask is an
independent lung cross-check and never replaces a mask.

Needs the pinned TotalSegmentator package installed and its task weights staged offline under
`third_party/weights/segmentation/totalsegmentator/`, plus the LungMask CLI and its exact
`R231.pth` checkpoint. TotalSegmentator is the canonical mask source; a low LungMask Dice flags
a case for review but never replaces the mask.

## 8. External RSPECT / RSNA-STR cohort

**Where:** `configs/runs/02_representation/rspect/single_task.yaml` and `multitask.yaml`
(`manifests/rsna_diagnosis.csv`)

A historical placeholder, not a verified contract. Confirm the real external manifest, its
patient-level split, and how its labels map onto this project's target names before running
either arm. Both are optional side branches.

## 9. CT-CLIP zero-shot

**Where:** `configs/runs/01_foundation/ct_clip_zero_shot.yaml` (status `unavailable`)

Needs all four of: an inspected text tower and tokenizer contract; a reviewed PE prompt set
frozen before evaluation; an explicit similarity-to-probability definition with a calibration
split; and a zero-shot evaluation entrypoint. Foundation materialization loads and profiles an
encoder — it does not classify, and must not be presented as a zero-shot result. Preflight
fails on `zero_shot.available` and `run.py` refuses to launch this arm.

## 10. Concept-bottleneck targets

**Where:** `configs/runs/90_deferred/concept_bottleneck/prognosis.yaml`
(`concept_bottleneck.enabled: false`)

Every enabled concept needs a real native / validated-silver / expert-reviewed column that is
also listed in `data.label_columns`. Embolic burden and vascular pruning are intentionally
absent: this dataset has no defensible target for either, and a bottleneck built on proxies
produces confident explanations of nothing.

## 11. Expert-validated CTPA segmentation

`data.segmentation` produces **public-model pseudo-anatomy**, not expert-validated CTPA
segmentation. A CTPA-specific fine-tuning/validation contract needs expert-reviewed
annotations, an inspected model/trainer factory and a frozen evaluation checkpoint; none
exist, so there is deliberately no such experiment in the active pipeline. Before adding one,
supply: an expert-reviewed annotation manifest, an inspected model/trainer factory and output
adapter, a public initialization checkpoint, and tuning restricted to train + validation.
`source/data/preflight.py` already validates that contract under the
`segmentation_validation` stage, and fails while any part of it is missing.

## What is deliberately absent

- No `tests/` directory and no pytest workflow in this repository yet.
- No `SEG02`, no segmentation fine-tuning, no supervised segmentation training.
- No per-model diagnosis code (`diagnosis_pretrained_model.py` and friends): the encoder
  checkpoint is a config variable, so one model serves every initialization.
- No batch/step limit for training stages. A training arm's scope is its dataset profile;
  inventing a `--max-cases` for it would create a pilot-only code path.
- No embolic-burden or vascular-pruning target.
- No zero-shot arm for CT-FM or TotalFM: they are image-only encoders.
- No implicit split creation, no test-split model selection, no fabricated labels, and silver
  labels are never treated as gold evaluation labels.
