# Blockers

Every unresolved real-data or checkpoint contract in one place. These are deliberate: the
configs refuse to guess a column name, a feature dimension or a checkpoint factory, so
preflight fails loudly instead of producing a number nobody can trust.

Nothing here is a code bug. Each item needs a decision or an artifact from outside the repo.

`python run.py plan <experiment>` marks a requirement `BLOCKED` when it maps to one of these,
and `MISSING` when the artifact simply has not been produced yet.

## 1. Public encoder contracts — blocks every image experiment

**Where:** `configs/components/backbone/ct_fm.yaml`, `ct_clip.yaml`, `totalfm.yaml`
**State:** `factory: ""`, `output_adapter: ""`, `feature_dim: 0`

Each file needs, from the actual inspected checkpoint:

- `factory` — import path that builds the model
- `output_adapter` — import path mapping upstream output to `ImageFeatures`
  (5-D `feature_map`, 2-D `global_embedding`)
- `feature_dim` — the real channel count of that feature map
- the weights staged locally under `third_party/weights/foundation/<backbone>/`
- SHA-256 and exact commit/revision recorded in `third_party/versions.yaml`

Do not copy values between backbones and do not infer them from repository names. TotalFM is
organ-patch based upstream, so its whole-volume pooling behaviour must be verified before its
probe means anything.

**Unblocks:** every foundation probe, every DAPT arm, alignment, both fixed probes, all
diagnosis and prognosis image arms, and everything downstream of them.

## 2. INSPECT manifests and the patient-level split

**Where:** `${PE_CLOUD_ROOT}/data/derived/manifests/`
**Needed:** `ctpa.csv`, `diagnosis.csv`, `prognosis.csv`, `paired_reports.csv`

Minimum columns: `patient_id`, `study_id`, `split`, `image_path`. `split` is one of
`train | validation | test | external`, and no patient may appear in two splits.

Create the split once, explicitly:

```bash
python tools/data/create_split.py --input SOURCE.csv --output manifests/ctpa.csv --create-split --seed 42
```

No split is ever created implicitly at train or evaluate time, and the test split is never
used to select checkpoints or thresholds.

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

## 6. Silver-label model weights

**Where:** `third_party/weights/silver/falcon/`, `third_party/weights/silver/medgemma/`

Complete local Hugging Face directories (config, tokenizer, weight shards, shard index).
Generation runs with `local_files_only: true`, so nothing is downloaded silently. Record the
exact model id, revision and SHA-256 in `third_party/versions.yaml`. SL02 loads both models on
every GPU rank — check VRAM before a full run.

## 7. Segmentation backend

**Where:** `configs/runs/00_data/segmentation.yaml`

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
- No embolic-burden or vascular-pruning target.
- No zero-shot arm for CT-FM or TotalFM: they are image-only encoders.
- No implicit split creation, no test-split model selection, no fabricated labels, and silver
  labels are never treated as gold evaluation labels.
