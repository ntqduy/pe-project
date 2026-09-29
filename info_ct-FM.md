# CT-FM frozen pipeline: preprocessing → diagnosis/prognosis

This document describes the current execution path in this repository. The raw INSPECT
release and the official `train/validation/test` assignment are read-only inputs. No step in
the CT-FM path creates a new split or moves a patient between splits.

## 1. Environment and storage

The scripts set these defaults automatically through `scripts/use_gcs_storage.sh`:

```text
PE_CLOUD_ROOT=/mnt/pe-storage
PE_RAW_INSPECT_ROOT=/mnt/Stanford_INSPECT_dataset
PE_DERIVED_ROOT=/mnt/pe-storage/derived
```

You only need to export these variables when your paths differ. Mount the bucket at
`/mnt/pe-storage` before running the scripts.

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

## 3. CT-FM feature cache and QC

The frozen CT-FM runs do not feed raw NIfTI or a squeezed volume to CT-FM during training.
`tools/data/build_ctfm_cache.py` applies the public CT-FM weights (SHA-256 checked) **once
per study**, at the upstream feature-extraction contract
(`third_party/repos/CT-FM/scripts/feature_extractor.py`):

```text
orientation       SPL
resample spacing  [3.0, 1.0, 1.0] mm (trilinear)
HU clipping       [-1024, 2048] -> [0, 1]
body crop         largest > 0 HU component
canvas            120 x 384 x 384 voxels = 360 x 384 x 384 mm (centre crop / air pad)
patches           non-overlapping 24 x 128 x 128 (5 x 3 x 3 = 45 per study)
stored tensor     float16 [513, 10, 24, 24]: 512 deepest CT-FM channels stitched from all
                  patches + 1 channel = fraction of each cell inside the body box
```

The previous cache squeezed the whole chest into one 24 x 128 x 128 tensor, i.e. ~12 x 1.9 x
2.6 mm voxels instead of the 3 x 1 x 1 mm CT-FM was pretrained on. Training reads the
cached tensors through `model.backbone=ct_fm_features`: the encoder only splits them. Its
`global_embedding` is the mean over cells inside the body (padding excluded), and
`extract_anatomy_features` uses it as the global branch, so canvas padding never enters the
model. Every pooled branch is then z-scored per channel with a centre/scale fit on the train
split and stored in the checkpoint (`organ_adapter.standardize_inputs`, see
`docs/05_diagnosis_training.md` "CT-FM frozen: pooling và chuẩn hoá feature"). ROI masks
for the anatomy arms are put on the same feature grid by world coordinates (fraction of
each cell covered). Consequences: CT-FM stays frozen (no fine-tuning/LoRA of CT-FM) and
image-space ROI counterfactuals are refused for these runs (§3.1 covers the runs that need them). Previews rebuild the CT canvas
from the raw NIfTI for the five preview patients.

A failed study is not reassigned to another split; it is recorded and dropped from the
CT-FM manifests only. The builder is resumable: studies whose features match the current
contract (fingerprint of spec + implementation + weight checksum) are reused.

```text
derived/datasets/<PROFILE>/manifests/ct_fm/      diagnosis / prognosis*.csv + README.txt:
                                                 the base manifests with image_path ->
                                                 the study's feature file (failed rows dropped)
derived/cache/<PROFILE>/ct_fm/
├── features/<study_id>.npy (+ .metadata.json)   tensor + spec, canvas geometry, feature-grid affine
├── aligned_masks/                                ROI masks on the feature grid (built on first use)
├── dataset.json                                  contract and aggregate QC
├── preprocessing_failures.csv                    only when studies fail
└── dropped_rows.csv                              only when rows are dropped
```

`dataset.json` holds cache coverage, failure/drop counts, tensor shapes, how many studies
were centre-cropped to the canvas and the body-coverage summary. These are preprocessing
QC, not task performance metrics.

Studies whose contract fingerprint changed are recomputed automatically. To force every
study to be recomputed anyway (`OVERWRITE=1` only replaces the run's outputs, not the cache):

```bash
PROFILE=smoke_30 REBUILD_CACHE=1 ACTION=prepare \
  bash scripts/run_ctfm_diagnosis.sh
```

### 3.1 Image-space CT-FM (`model.backbone=ct_fm`)

The fine-tuned runs (LoRA diagnosis/prognosis matrices, ROI students, counterfactuals, DAPT,
alignment) need CT-FM to see an image, so they cannot use the feature cache. They read the
shared volume cache (`cache/<PROFILE>/volumes`, 128^3, RAS, [-1000, 1000] HU min-max) and the
encoder converts each batch to the CT-FM contract before the first convolution
(`source/components/encoders/image/ct_fm.py`, `factory_kwargs.input_orientation` /
`input_hu_range` in `configs/components/backbones.yaml`):

```text
orientation   RAS [x, y, z] -> SPL [z, -y, -x] (checked equal to nibabel reorientation);
              every pyramid level is turned back to RAS, so ROI masks still line up
intensity     v -> (2000 v - 1000 + 1024) / 3072, i.e. CT-FM's [-1024, 2048] scaling;
              HU above 1000 was clipped by the cache and cannot be recovered
spacing       NOT converted: ~2.4 x 2.4 x 2.0 mm instead of 3 x 1 x 1 mm
```

The dataset checks every volume's sidecar (`orientation`, `hu_range`, `normalization`)
against these values and stops on a mismatch; `preprocessing.window` is refused for these
runs because it would rescale intensities the encoder already converts. The weights still
load strictly (161/161 tensors): the conversion adds no parameters.

LoRA: SegResEncoder has only convolutions, so `peft.target_modules` (`attn`, `projection`,
transformer names) matches nothing. The registry entry sets
`lora_target_modules: [layers.3.blocks, layers.4.blocks]`, which takes precedence: 16 Conv3d
layers get a rank-8 conv update (same kernel/stride as the frozen conv, 1x1 back, zero
init), 1.38 M trainable parameters. `peft_report` records `target_source: backbone`.

Feeding these runs at the true 3 x 1 x 1 mm contract would need a second image cache and
sliding-window training. Measured on the L4 (bf16, 45 patches of 24 x 128 x 128 per study):
0.42 s/study per epoch for LoRA forward+backward versus 0.06 s/study for the current 128^3
input (~7x), plus a 120 x 384 x 384 int16 volume per study (35 MB, ~0.8 TB for 23,248
studies, versus 8 MB now). Not implemented.

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
manifests/ct_fm/diagnosis.csv
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
├── logs.txt                      # log terminal: train rồi evaluate (ghi nối tiếp)
├── history.csv                   # metric train/validation theo từng epoch
├── result.csv                    # metric từng split (train, validation, test) cho mỗi target
├── predictions.csv               # y_true / y_prob / y_pred từng ca, mọi split
├── training_curves.png           # loss + AUROC train/val theo epoch, vạch best.ckpt
└── preview/
    ├── NN_<patient>_<study>_<TP|TN|FP|FN>.html  # viewer Grad-CAM: mọi slice input, CT | CT + CAM
    └── NN_<patient>_<study>_<TP|TN|FP|FN>.png   # header + montage 8 slice CAM cao nhất
```

Root chỉ giữ `resolved_config.yaml` và `result.json`; các bản sao checkpoint, log, metrics,
environment và file QC trung gian được dọn sau khi run hoàn tất. Preview dựng lại canvas CT
120×384×384 (3×1×1 mm, SPL) từ NIfTI gốc theo sidecar feature, hiển thị phía trước ở trên, trái
bệnh nhân bên phải ảnh, và vẽ Grad-CAM của logit target chính tại feature map CT-FM
512×10×24×24 (mỗi ô 36×16×16 mm). CAM được chuẩn hóa một lần cho cả volume; slice gốc NIfTI
được tính theo chuỗi preprocessing. Nếu không có gradient hoặc CAM toàn 0/NaN, preview ghi rõ
và không vẽ heatmap, không thay bằng feature activation.

Preview mặc định lấy 5 patient đầu tiên của validation manifest để kiểm tra định tính;
không dùng test để tạo heatmap và không làm thay đổi split. Với model prognosis chỉ có
clinical branch, preview sẽ được đánh dấu `skipped`. Cách đọc, kiểm tra căn chỉnh và giới hạn:
`docs/05_diagnosis_training.md`, mục "Preview Grad-CAM"; tạo lại preview cho run đã evaluate:
`python tools/tasks/gradcam_preview.py --run-dir <run>`.

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
    ├── predictions.csv
    ├── training_curves.png
    └── preview/
```

`epoch_<epochs_run>/result.csv` có một dòng cho mỗi (target, split), cùng cột cho mọi run
(kể cả zero-shot PENet) để ghép và so sánh; mô tả cột, `threshold`, `threshold_rule` và cột
`note` nằm trong `docs/09_output_reference.md`:

```text
split=train                    point metrics, không bootstrap CI
split=validation               point metrics dùng threshold chọn trên validation
split=test                     point metrics + patient-bootstrap CI (AUROC, AUPRC)
```

Với prognosis, `result.csv` có dòng cho toàn bộ target trong config, hiện gồm:
`1_month_mortality`, `6_month_mortality`, `12_month_mortality`, `1_month_readmission`,
`6_month_readmission`, `12_month_readmission`, `12_month_PH`, cộng thêm hai cột
`calibration_intercept`, `calibration_slope`. Cột `experiment` (output ID) phân biệt hai run
`prognosis_all_patient.csv` và `prognosis_pe_positive.csv`; `cohort` nằm trong `result.json`.
Không gộp bệnh nhân PE vào cohort all-patient.

`result.json` chứa metric của mọi split, `ci_low`, `ci_high`, `valid_replicates` của mọi
metric, và số ca dương/âm (`<split>_positives`, `<split>_negatives`). Nó không chứa CT
preprocessing QC; CT-QC nằm trong artifact của CT-FM cache ở trên.

Với smoke run (vd. `smoke_30`: train chỉ 1/10 ca PE dương, test 1/10), nhiều ô trong
`result.csv` sẽ trống và `note` ghi lý do (1 lớp, <2 bệnh nhân, không có dự đoán âm, ...).
Đó là giới hạn của dữ liệu smoke, không phải lỗi tính toán; dùng official test split đầy đủ
cho kết quả khoa học. Calibration slope/intercept chỉ được tính khi mỗi lớp có ít nhất
`evaluation.calibration_min_events` (mặc định 10) ca và hai lớp không bị tách hoàn toàn.

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
