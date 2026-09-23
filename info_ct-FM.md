# CT-FM frozen pipeline: preprocessing → diagnosis/prognosis

This document describes the current execution path in this repository. The raw INSPECT
release and the official `train/validation/test` assignment are read-only inputs. No step in
the CT-FM path creates a new split or moves a patient between splits.

## 1. Environment and storage

The scripts expect the storage roots to be exported before running:

```bash
export PE_CLOUD_ROOT=/mnt/pe-storage
export PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset
export PE_DERIVED_ROOT=/mnt/pe-storage/derived
export PE_LOCAL_CACHE_ROOT=/mnt/pe-project/cache
```

The raw release must contain the CTPA volumes below:

```text
$PE_RAW_INSPECT_ROOT/CT/full/CTPA/
```

Derived data is written under:

```text
$PE_DERIVED_ROOT/datasets/<PROFILE>/
```

The persistent model output is written under:

```text
$PE_CLOUD_ROOT/pe-project/outputs/
```

## 2. Stage 0: INSPECT preprocessing

Run:

```bash
PROFILE=smoke_30 bash scripts/run_preprocessing.sh
```

For a larger rehearsal or the full cohort:

```bash
PROFILE=test_500_sample bash scripts/run_preprocessing.sh
PROFILE=full_inspect bash scripts/run_preprocessing.sh
```

The shell script delegates to the dataset CLI and the shared implementation:

```text
scripts/run_preprocessing.sh
  → run.py run data.dataset.<PROFILE>
  → tools/data/build_dataset.py
  → source/data_preprocessing/pipeline.py
```

The shared preprocessing order is:

```text
raw INSPECT tables
  → source join
  → eligibility/noise filters
  → missing/corrupt volume integrity checks
  → patient-level label adjudication
  → patient-level sampling inside the existing official split
  → split/leakage audit
  → EHR readiness and profiles
  → sPESI artifacts
  → diagnosis/prognosis manifests
  → CT volume cache and provenance sidecars
```

The preprocessing package is [`source/data_preprocessing/`](/mnt/pe-project/source/data_preprocessing).
It treats the raw release as read-only. The resulting profile normally contains:

```text
derived/datasets/<PROFILE>/
├── manifests/
│   ├── diagnosis.csv
│   ├── prognosis.csv
│   ├── prognosis_all_patient.csv
│   └── prognosis_pe_positive.csv
├── volumes/
├── clinical/
├── audit/
├── data_quality.json
├── data_quality.md
└── dataset.json
```

Use preflight without starting a long build:

```bash
PROFILE=full_inspect ACTION=preflight bash scripts/run_preprocessing.sh
```

## 3. CT-FM-specific cache and QC

The task does not feed raw NIfTI directly to CT-FM. The existing official-split manifests
are adapted into a CT-FM cache by:

```text
tools/data/build_ctfm_cache.py
```

The cache contract currently records:

```text
orientation       SPL
resample spacing  [3.0, 1.0, 1.0] mm
HU clipping       [-1024, 2048]
normalization     minmax
target shape      [24, 128, 128]
dtype             float32
```

The cache builder reads all existing source manifests, preprocesses each study, then calls
`validate_cache_entry()` before including the study in a CT-FM manifest. A failed volume is
not reassigned to another split. It is recorded and excluded from the CT-FM-derived manifest.

The CT-QC artifacts are deliberately separate from diagnosis/prognosis clinical metrics:

```text
derived/datasets/<PROFILE>/ct_fm_frozen/
├── manifests/
├── volumes/
├── dataset.json                    # CT contract and aggregate QC
├── preprocessing_failures.csv      # study-level preprocessing/validation failures
└── dropped_rows.csv                # source rows excluded because their CT failed
```

`ct_fm_frozen/dataset.json` contains cache coverage, failure/drop counts, output shape and
spacing distributions, dtype counts, crop-fraction summary, and the preprocessing
fingerprint. These are preprocessing QC, not task performance metrics.

To force regeneration after changing the cache implementation:

```bash
PROFILE=smoke_30 OVERWRITE=1 ACTION=prepare \
  bash scripts/run_ctfm_diagnosis.sh
```

## 4. Diagnosis: CT-FM frozen + MLP

Run the diagnosis wrapper:

```bash
PROFILE=smoke_30 EPOCHS=1 GPUS=0 ACTION=all \
  bash scripts/run_ctfm_diagnosis.sh
```

The full run is:

```bash
PROFILE=full_inspect EPOCHS=50 EARLY_STOPPING=10 GPUS=0,1 ACTION=all \
  bash scripts/run_ctfm_diagnosis.sh
```

## 9. Mask, silver-label, and anatomy proposal flow

The stages are intentionally separate. `run_preprocessing.sh` builds the filtered official
split manifests and CT cache; it does not silently run segmentation or silver-label models.
Run the data stages in this order:

```bash
PROFILE=smoke_30 bash scripts/run_preprocessing.sh
PROFILE=smoke_30 GPUS=0 bash scripts/run_segmentation.sh
PROFILE=smoke_30 ROI_WORKERS=4 bash scripts/run_roi.sh
PROFILE=smoke_30 MAX_REPORTS=30 GPUS=0 bash scripts/run_silver_labels.sh
```

After the smoke artifacts pass QC, repeat with `PROFILE=full_inspect`. The baseline task
wrappers then run CT-FM pretrained/frozen plus an MLP using the existing official manifests:

```bash
PROFILE=full_inspect GPUS=0,1 bash scripts/run_ctfm_diagnosis.sh
PROFILE=full_inspect GPUS=0,1 bash scripts/run_ctfm_prognosis_all.sh
PROFILE=full_inspect GPUS=0,1 bash scripts/run_ctfm_prognosis_pe.sh
```

The proposal wrapper uses the same frozen CT-FM encoder, adds trainable heart/PA/lung organ
adapters, and compares feature concatenation against a learned Soft-MoE router:

```bash
TASK=diagnosis FUSION=concat PROFILE=full_inspect GPUS=0,1 bash scripts/run_ctfm_anatomy.sh
TASK=diagnosis FUSION=moe PROFILE=full_inspect GPUS=0,1 bash scripts/run_ctfm_anatomy.sh
TASK=prognosis COHORT=pe FUSION=concat PROFILE=full_inspect GPUS=0,1 bash scripts/run_ctfm_anatomy.sh
TASK=prognosis COHORT=pe FUSION=moe PROFILE=full_inspect GPUS=0,1 bash scripts/run_ctfm_anatomy.sh
```

All task launchers use `torchrun` when multiple GPUs are specified. No stage reassigns a
patient between train, validation, and test; filtering preserves the original split and
records dropped rows in the audit artifacts.

The execution path is:

```text
scripts/run_ctfm_diagnosis.sh
  → scripts/run_ctfm_frozen.sh
  → tools/data/build_ctfm_cache.py       [prepare, if needed]
  → tools/tasks/train_task.py             [train]
  → source/engine/factory.py              [build CT-FM + task head]
  → source/engine/trainer.py              [MLP training]
  → tools/tasks/evaluate.py               [validation threshold + test metrics]
```

The diagnosis config is:

```text
configs/runs/01_foundation/ct_fm_frozen_diagnosis.yaml
```

Its contract is:

```text
backbone       CT-FM
encoder        frozen
trainable      classification head/adapter allowed by the resolved config
architecture   concat_mlp
target         pe_present
```

The official split is read from:

```text
ct_fm_frozen/manifests/diagnosis.csv
```

Training uses the train and validation portions only. Test is not used for optimization,
threshold tuning, or early stopping.

Mỗi epoch ghi thêm `train_auroc`, `val_auroc`, `train_auprc` và `val_auprc` vào history.
Các AUC này chỉ để theo dõi đường cong training; checkpoint vẫn được chọn bằng
`negative_validation_loss`, không chọn bằng test metric.

## 5. Prognosis: all patients and PE-positive patients

All eligible patients:

```bash
PROFILE=smoke_30 EPOCHS=1 GPUS=0 ACTION=all \
  bash scripts/run_ctfm_prognosis_all.sh
```

PE-positive cohort only:

```bash
PROFILE=smoke_30 EPOCHS=1 GPUS=0 ACTION=all \
  bash scripts/run_ctfm_prognosis_pe.sh
```

Full run on two GPUs:

```bash
PROFILE=full_inspect EPOCHS=50 EARLY_STOPPING=10 GPUS=0,1 ACTION=all \
  bash scripts/run_ctfm_prognosis_all.sh

PROFILE=full_inspect EPOCHS=50 EARLY_STOPPING=10 GPUS=0,1 ACTION=all \
  bash scripts/run_ctfm_prognosis_pe.sh
```

The prognosis configs are:

```text
configs/runs/01_foundation/ct_fm_frozen_prognosis_all.yaml
configs/runs/01_foundation/ct_fm_frozen_prognosis_pe.yaml
```

By default, prognosis is multi-task over these seven endpoints:

```text
1_month_mortality
6_month_mortality
12_month_mortality
1_month_readmission
6_month_readmission
12_month_readmission
12_month_PH
```

To train/evaluate one endpoint independently, set `TARGET`. This creates a distinct output
ID and does not modify the source manifest:

```bash
TARGET=12_month_PH PROFILE=full_inspect EPOCHS=50 GPUS=0,1 \
  bash scripts/run_ctfm_prognosis_pe.sh
```

Censored or unavailable outcome labels remain in the manifest for cohort/split accounting,
but are masked during training and excluded from the corresponding evaluation metric.

## 6. Training controls

Each wrapper accepts:

```text
ACTION=prepare|train|evaluate|all|preflight|dry
PROFILE=smoke_30|test_500_sample|full_inspect
GPUS=0 or 0,1,2,3
EPOCHS=1 or 50
EARLY_STOPPING=10
BATCH_SIZE=<optional override>
OVERWRITE=1
TARGET=<prognosis endpoint only>
```

Early stopping is based on the validation objective `negative_validation_loss`. The selected
checkpoint is written to `best.ckpt`; the final state is written to `last.ckpt`. Task training
does not calculate confidence intervals; CI/bootstrap chỉ được tính trong bước evaluate trên
test split.

Sau khi train/evaluate xong, mỗi run chỉ giữ metadata tối thiểu ở root và một bundle theo số
epoch thực tế đã chạy:

```text
outputs/<family>/<RUN_ID>/epoch_<epochs_run>/
├── checkpoint/
│   ├── best.ckpt                 # validation-selected checkpoint
│   └── last.ckpt                 # state ở epoch cuối, kể cả khi early stopping
├── logs.txt                      # terminal stream của wrapper + run log
├── result.csv                   # một dòng train, validation, test cho mỗi target
├── training_curves.pdf           # train/val loss và train/val AUROC
└── preview/
    ├── *_axial_feature_activation.png   # mean absolute activation, axial middle slice
    ├── *_axial_gradcam.png              # target-specific axial heatmap
    ├── *_coronal_feature_activation.png # mean absolute activation, vertical coronal slice
    ├── *_coronal_gradcam.png            # target-specific vertical coronal heatmap
    ├── README.txt                       # preview contract và lỗi từng sample nếu có
    └── summary.json              # số sample đã ghi và số lỗi
```

Root chỉ giữ `resolved_config.yaml` và `result.json`; các bản sao checkpoint, log, metrics,
environment và file QC trung gian được dọn sau khi run hoàn tất. Với CT-FM frozen, nếu
gradient backbone không khả dụng thì ảnh Grad-CAM dùng tên
`*_axial_feature_activation_max.png` và `*_coronal_feature_activation_max.png`.

Preview mặc định lấy 5 patient đầu tiên của validation manifest để kiểm tra định tính;
không dùng test để tạo heatmap và không làm thay đổi split. Với model prognosis chỉ có
clinical branch, preview sẽ được đánh dấu `skipped`; CT-FM image branch thì có cả activation
map và Grad-CAM theo target chính.

## 7. Task metrics, CI, and patient bootstrap

Clinical task metrics are generated by:

```text
tools/tasks/evaluate.py
```

The evaluation protocol is:

```text
1. Load best.ckpt.
2. Run validation inference.
3. Select the operating threshold on validation only (Youden by default).
4. Run test inference with the fixed validation threshold.
5. Compute point metrics on test.
6. Run patient-level bootstrap on test predictions.
7. Save point estimate, 95% CI, and bootstrap metadata.
```

Default bootstrap settings in the diagnosis/prognosis contracts are:

```text
unit        patient
replicates  2000
confidence  0.95
```

Diagnosis metrics include AUROC, AUPRC, accuracy, balanced accuracy, sensitivity,
specificity, PPV, NPV, F1, and Brier score. Prognosis includes these classification metrics
plus calibration intercept and calibration slope; a calibration curve is also written.

Task output is written under the relevant persistent output family:

```text
outputs/diagnosis/<RUN_ID>/
outputs/prognosis/<RUN_ID>/
├── resolved_config.yaml
├── result.json
└── epoch_<epochs_run>/
    ├── checkpoint/{best.ckpt,last.ckpt}
    ├── logs.txt
    ├── result.csv
    ├── training_curves.pdf
    └── preview/
```

`epoch_<epochs_run>/result.csv` là bảng wide-format; mỗi dòng là một split/target:

```text
split=train                    point metrics, không bootstrap CI
split=validation               point metrics dùng threshold chọn trên validation
split=test                     point metrics + patient-bootstrap CI
```

Với prognosis, `final_metric` được ghi cho toàn bộ target có trong config, hiện gồm:
`1_month_mortality`, `6_month_mortality`, `12_month_mortality`, `1_month_readmission`,
`6_month_readmission`, `12_month_readmission`, `12_month_PH`. Mỗi dòng giữ `cohort`, `target`,
`split`, `metric`, `value`, `ci_low`, `ci_high`, `valid_replicates`. Vì vậy hai run
`prognosis_all_patient.csv` và `prognosis_pe_positive.csv` được phân biệt rõ bằng `cohort` và
output ID; không gộp bệnh nhân PE vào cohort all-patient.

`metrics.json` contains task metrics and their `ci_low`, `ci_high`, and
`valid_replicates`. It does not contain CT preprocessing QC. CT-QC remains in the separate
CT-FM cache artifacts described above.

For a one-epoch smoke run, CI can be unavailable when the test subset has fewer than two
patients or lacks both outcome classes. That is expected for a technical rehearsal; use the
full official test split for the scientific result.

## 8. Recommended order before the full experiment

```bash
# A. Validate storage, dependencies, raw release, and existing metadata.
PROFILE=smoke_30 ACTION=preflight bash scripts/run_preprocessing.sh

# B. Build a small official-split profile.
PROFILE=smoke_30 bash scripts/run_preprocessing.sh

# C. Build and validate the CT-FM cache.
PROFILE=smoke_30 ACTION=prepare bash scripts/run_ctfm_diagnosis.sh

# D. Run one diagnosis epoch and inspect logs/result artifacts.
PROFILE=smoke_30 EPOCHS=1 GPUS=0 ACTION=all \
  bash scripts/run_ctfm_diagnosis.sh

# E. Repeat the one-epoch check for prognosis all/PE.
PROFILE=smoke_30 EPOCHS=1 GPUS=0 ACTION=all \
  bash scripts/run_ctfm_prognosis_all.sh
PROFILE=smoke_30 EPOCHS=1 GPUS=0 ACTION=all \
  bash scripts/run_ctfm_prognosis_pe.sh

# F. Only after the smoke artifacts and CI behavior are acceptable, scale up.
PROFILE=full_inspect EPOCHS=50 EARLY_STOPPING=10 GPUS=0,1 ACTION=all \
  bash scripts/run_ctfm_diagnosis.sh
```
