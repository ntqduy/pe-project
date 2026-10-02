# 01. Data pipeline

**Đọc khi:** cần biết cohort được dựng thế nào từ bản phát hành INSPECT, mask giải phẫu / ROI /
silver label được sinh ra sao, split train/validation/test được bảo vệ thế nào, và một batch
khi train chứa gì.

**Code chính:** `source/data/` (`build/pipeline.py`, `dataset.py`, `paths.py`, `experiment_splits.py`,
`preflight.py`, `profiles/`), `source/segmentation/`, `source/roi/`, `source/silver/`,
`tools/data/`, `tools/create_masks/generate_masks.py`, `tools/build_rois/build_rois.py`,
`tools/silver_labels/generate_silver_labels.py`, `configs/runs/00_data/`, `scripts/data/`.

Tổng quan ở [README.md](README.md); model dùng các output này ở [02_models.md](02_models.md);
lệnh chạy và cây output đầy đủ ở [05_running_outputs.md](05_running_outputs.md).


---

## 1. Dataset (stage 0)

### 1.1 Ba profile

`ACTIVE_PROFILES = ("smoke_30", "test_500_sample", "full_inspect")` trong
`source/data/profiles/__init__.py`. Cả ba kế thừa `_common.yaml` (eligibility, integrity,
adjudication, manifests, preprocessing, cache_qc, ehr, spesi); chỉ khác khối `sampling`.

| Profile | `sampling` | Mục đích |
|---|---|---|
| `smoke_30` | seed `20260703`, 10 patient mỗi split, lấy **trước** eligibility/CT check | Kiểm tra kỹ thuật code path. Cohort cuối có thể < 30 patient. Không dùng để chọn hay báo cáo model |
| `test_500_sample` | seed `20260704`, `total_patients: 500` sau khi làm sạch | Rehearsal: chia theo tỉ lệ kích thước từng official split |
| `full_inspect` | tắt | Toàn bộ cohort đủ điều kiện, giữ official split |


**Profile là data, không phải code:** `build_dataset` là implementation duy nhất.
`assert_shared_preprocessing()` so `PreprocessingSpec.fingerprint()` của các profile và raise
`DatasetProfileError` nếu khác nhau (preflight check `shared_preprocessing`), nên không profile
nào lặng lẽ đổi preprocessing làm kết quả mất tính so sánh. Config run
`configs/runs/00_data/dataset/<profile>.yaml` chỉ chọn profile.

### 1.2 Luồng `build_dataset`

```text
InspectSource.records()                      sources.py      join các bảng TSV của release
  -> [smoke_30] sample_patients trước        sampling.py
  -> governance exclusion                    leakage.py
  -> apply_eligibility                       filters.py      ledger loại trừ
  -> check_volumes                           integrity.py    CT thiếu/hỏng
  -> patient_labels                          adjudication.py
  -> [test_500_sample] sample_patients       sampling.py
  -> audit_split_integrity + require_no_leakage     (lần 1, trước khi ghi cache)
  -> preprocess_study + validate_cache_entry volumes.py      cache .npy + sidecar
  -> audit_split_integrity + require_no_leakage     (lần 2, cohort cuối)
  -> build_ehr_profiles (ehr.py) -> build_spesi_artifacts (spesi.py)
  -> build_manifests (manifest_writer.py) -> exclusions.csv, data_quality.md, dataset.json
```


**Đọc release (`sources.py`).** Release root `<raw_inspect>/CT/full` chứa `splits_*`, `labels_*`,
`study_mapping_*`, `series_metadata_*`, `study_metadata_*`, `impressions_*` (glob file mới nhất)
và `CTPA/<image_id>.nii.gz`. Các điểm đáng nhớ:
- Split `valid` đổi tên thành `validation`; study không có trong bảng split bị bỏ; person_id lệch
  giữa mapping và bảng split → raise.
- `_primary_series` chọn series nhiều lát nhất (hoà thì lát mỏng hơn), không lấy dòng đầu: lấy
  dòng đầu đọc geometry của localizer và làm ~1/5 release thành study "1 lát".
- `official_splits()` đọc lại bảng split **độc lập** với `records()`, để leakage audit so cohort
  cuối với release chứ không so một giá trị với chính nó.

**Eligibility (`filters.py`).** Registry governance loại trước (`excluded_patient_registry`), rồi
các rule theo `RULE_ORDER`; rule fail đầu tiên sở hữu việc loại trừ, ghi vào
`manifests/exclusions.csv`. Ngưỡng chính trong `_common.yaml`: `minimum_slices: 2`,
`maximum_slice_thickness_mm: 5.0`, `maximum_pixel_spacing_mm: 2.0`, `allowed_modalities: [CT]`,
yêu cầu có file CT / report / labels / series metadata. `require_ehr_crosswalk: false` (crosswalk
chỉ cần cho prognosis). Danh sách rule đầy đủ: `RULE_ORDER` trong `filters.py`.

**Nhãn cấp patient (`adjudication.py`).** `adjudicate_binary` gộp nhãn nhiều study theo thứ tự
`TRUE > CENSORED > FALSE > MISSING` (CENSORED thắng FALSE vì censored không phải âm tính quan sát
được). Nhãn cấp patient chỉ dùng để **stratify sampling**; manifest vẫn ghi nhãn cấp study.
`first_index_study` chọn study sớm nhất làm index cho prognosis; CENSORED/MISSING không bao giờ
thành "sống". Đây là nhãn gốc INSPECT, không phải silver label (§4).

**Leakage (`leakage.py`).** `audit_split_integrity` báo lỗi khi patient ở > 1 split, `study_id`
trùng, study đổi split so với release, hoặc (khi `require_all_splits`) thiếu hẳn một split;
`require_no_leakage` raise `LeakageError`. Chạy hai lần (trước cache và cohort cuối).

**Chuẩn hoá CT (`volumes.py`, `ctpa-physical-v2`).** Giá trị trong `_common.yaml`:

| Bước | Giá trị |
|---|---|
| Reorient / Resample (linear) | `RAS`, `[1.5, 1.5, 1.5]` mm |
| Clip HU | `[-1000, 1000]` |
| Crop body | `external_body`: component lớn nhất có HU > -900, margin 10 mm |
| Về shape cố định | `fit` vào `[128, 128, 128]` (zoom cả vùng crop, spacing hiệu dụng mỗi case khác nhau và được ghi lại; không centre-crop để khỏi mất giải phẫu trên/dưới) |
| Chuẩn hoá | `minmax` về [0,1], `float32`, lưu `npy` |
| Patch grid | `[64,64,64]`, overlap 0.5 |

`preprocess_study` ghi atomically `<study_id>.npy`, sidecar `.npy.metadata.json` (affine gốc /
reorient / resample / output, spacing, crop box, fingerprint, SHA-256 nguồn) và `.npy.patches.csv`.
Cache cũ chỉ tái dùng khi fingerprint, implementation, đường dẫn và SHA-256 khớp, nếu không raise.
`validate_cache_entry` (khi `cache_qc.enabled`) kiểm tra shape, dtype, NaN/Inf, khoảng [0,1],
sidecar; lỗi → study bị loại (`ct_preprocessing_failed`, `ct_cache_qc_failed`). Sidecar quan trọng
ở runtime: mask nằm trong geometry CT gốc còn input là grid đã crop/resample, nên `CTPADataset`
dùng `output_affine` để map mask (§1.8).

### 1.3 EHR an toàn với leakage (`ehr.py`)

- Giải nén `meds_omop_inspect.tar.gz` **một lần** vào `<cache_root>/inspect_meds_omop/` (chặn
  symlink và path traversal); archive thô không được copy vào dataset.
- Chỉ study có `has_ehr_crosswalk` mới join EHR; study khác ở lại cohort với EHR missing.
- **Mốc thời gian:** event tính chỉ khi `event_time < procedure_datetime - end_before_ctpa_hours`.
  Event không có thời gian, bảng `note`, hoặc code mô tả khớp CTPA / CT pulmonary angiography /
  radiology report bị loại. Hai profile: `EHR_0_h` (mặc định) và `EHR_24_h` (đệm 24 h trước
  CTPA); cả hai đều không bao giờ nhận event sau CTPA.

### 1.4 sPESI (`spesi.py`, `source/clinical/spesi.py`)

Mỗi tiêu chí 1 điểm, ≥ 1 điểm là high risk: tuổi > 80, ung thư, bệnh tim phổi mạn, mạch ≥ 110,
huyết áp tâm thu < 100, SpO2 < 90. Mã nguồn nằm ở `configs/clinical/spesi_mapping.yaml`.
Quy tắc: chỉ event **trước** CTPA (`event_window_after_index_hours: 0`; INSPECT gốc cho phép
+48 h), vital lấy giá trị mới nhất trong 240 h, kiểm tra đơn vị và khoảng sinh lý; comorbidity
không có code = âm tính. Thiếu bất kỳ thành phần nào → **không chấm điểm** (`spesi_computable=0`),
không cộng điểm từng phần. Nếu EHR không `built`, sPESI có status `unavailable` (không làm hỏng build). `spesi_evaluable.csv` là subset có điểm để mọi arm so trên cùng tập.

> **Lưu ý khoa học, cancer "trọn đời":** `comorbidity_lookback_hours: null` nghĩa là bất kỳ code
> ung thư nào trước CTPA, ở bất kỳ thời điểm nào, cũng cho 1 điểm (prefix `ICD10CM/C` gồm cả
> chẩn đoán cũ). sPESI lâm sàng hỏi về ung thư đang hoạt động, nên `cancer` và do đó
> `spesi_high_risk` có thể bị ước lượng cao hơn. Cùng quy tắc áp cho `cardiopulmonary_disease`.


### 1.5 Manifests (`manifest_writer.py`)

Ghi vào `<dataset>/manifests/`. Cột định danh luôn có: `patient_id, study_id, split, image_path`
(khi `dataset.preprocess: true`, `image_path` trỏ tới `.npy` trong cache).

| File | Nội dung |
|---|---|
| `ctpa.csv` | mọi study đủ điều kiện; input của segmentation và ROI |
| `diagnosis.csv` | mọi study + nhãn PE gốc (`pe_present, pe_acute, pe_subsegmental_only, ...`) |
| `prognosis.csv`, `prognosis_all_patient.csv`, `prognosis_pe_positive.csv` | index study đầu của patient (legacy: PE dương + acute / mọi patient / PE dương); kèm `prognosis_cohort_membership.csv` ghi patient thuộc cohort nào. Mỗi outcome (`1/6/12_month_mortality`, `..._readmission`, `12_month_PH`) có các cột `<o>, _status, _observed, _censored, _time_to_event` |
| `reports.csv` / `paired_reports.csv` | text report (cho silver label) / cặp ảnh-impression |
| `exclusions.csv` | ledger loại trừ: `rule, detail` |

- Nhãn mã hoá `1`/`0`; CENSORED/MISSING để **rỗng** (dataset coi là không có nhãn, không phải 0).
- Mọi manifest gắn cột clinical (EHR của từng profile, `spesi, spesi_high_risk, spesi_computable`, cờ modality `has_ehr, has_spesi`, ...).
- `data_quality.md` (`quality.py`): cohort funnel, phân bố nhãn theo task, readiness của
  index-time / EHR / sPESI, sampling và leakage guard. Mở file này và `exclusions.csv` trước khi train.

### 1.6 Output stage 0, CT-FM cache, fold/fraction

**CT-FM feature cache (`tools/data/build_ctfm_cache.py`).** Chạy sau stage 0, không tạo split,
kiểm tra lại split và nhãn. Preprocessing riêng theo contract CT-FM upstream: đọc **NIfTI gốc**,
`SPL`, spacing 3×1×1 mm, clip `[-1024, 2048]` HU, canvas `120×384×384`, patch `24×128×128` không
chồng lấn → SegResEncoder. Mỗi study: `features/<study_id>.npy` float16 `[513, d, h, w]` (512
kênh + 1 kênh "tỉ lệ ô nằm trong body") và `pooled/<study_id>.npy` `[513,1,1,1]`. Ghi
`manifests/ct_fm/*.csv` (`image_path` → feature file, thêm `pooled_path`; study lỗi vào
`dropped_rows.csv`). Chi tiết dùng ở [02_models.md](02_models.md).

**Fold và training-fraction (`experiment_splits.py`, `tools/data/build_split_manifests.py`).**
Tạo một assignment cấp patient được lưu lại: pool = patient official `train` + `validation`; trong
mỗi nhóm label, shuffle theo `seed` rồi chia vòng tròn vào `cv_fold`; `fraction_rank` đều trong
[0,1) theo label nên fraction được stratify và lồng nhau (25% ⊂ 50% ⊂ 75% ⊂ 100%). Patient official
`test` **luôn giữ nguyên**; fold k = validation là `cv_fold == k`, phần còn lại là train; fraction
chỉ cắt **train**. `base_fingerprint` chặn dùng assignment cũ khi manifest gốc đã đổi. Tên:
`<task>_<label>_k<folds>_s<seed>`, thư mục `<official|fold{k}>/frac{PPP}/`. `parse_fraction`: "25"
hay "12.5%" là phần trăm, "0.25" là phân số, "1" bị từ chối vì mơ hồ.

### 1.7 Bảo đảm split cấp patient

1. Split lấy nguyên từ bảng release; không bước nào tạo split mới.
2. Mapping và bảng split phải cùng person_id; một patient không ở hai split.
3. Governance exclusion và sampling theo **patient**; patient được chọn giữ mọi study; sampling
   rút độc lập trong từng official split.
4. `audit_split_integrity` chạy 2 lần, so với `official_splits()` đọc độc lập.
5. Fold/fraction không chạm test; chỉ chia lại pool train+validation theo patient.
6. Vocabulary EHR chỉ đếm trên train; EHR không nhìn event sau CTPA.
7. `build_ctfm_cache.py` và preflight (`audit_manifest`, `patient_overlap`) kiểm tra lại overlap.

### 1.8 Runtime: `ProjectPaths`, `CTPADataset`, preflight

**`ProjectPaths` (`source/data/paths.py`).** Đọc khối `paths` (`configs/paths.yaml`) và biến môi
trường. Chỉ `PE_RAW_INSPECT_ROOT` / `PE_DERIVED_ROOT` thắng giá trị YAML; `cache_root` lấy từ
`PE_LOCAL_CACHE_ROOT` hoặc `<code_root>/cache`. `output_root` bắt buộc bằng

**`CTPADataset` (`source/data/dataset.py`).** Khởi tạo: đọc manifest, lọc `split`; prognosis bỏ
dòng không có outcome quan sát được; nếu có `roi_mask_ids`, đọc `roi_manifest` và chỉ giữ dòng
`roi_id`/`control_for` khớp, `status ∈ {PASS, SUSPICIOUS, pass}`, có `roi_path`; nếu có
`silver_path`, chỉ lấy silver `accepted`. Một item:

| Key | Shape / kiểu | Ghi chú |
|---|---|---|
| `volume` | `[1,x,y,z]` (batch `[B,1,x,y,z]`) | cache dense: RAS, 128³, [0,1]; CT-FM: `[513,d,h,w]` hoặc pooled `[513,1,1,1]` |
| `masks` | `{region: [1,x,y,z]}` | mask NIfTI map lên grid input theo `output_affine` của sidecar (`source/imaging/grid.py`, nearest; cache ở `aligned_masks/`) |
| `labels`, `label_valid` | `[n_labels]`, bool | ô rỗng → NaN → `label_valid=False` (bị mask khỏi loss) |
| `ehr`, `spesi` (+ `*_available`) | `[n]`, bool | theo `data.ehr_columns` / `data.spesi_columns` |
| `silver_labels`, `silver_valid` | `[n_targets]`, bool | khi có silver |

`input_contract` (nếu encoder yêu cầu) so field trong sidecar (orientation, HU range), không khớp
→ `VolumeLoadError`. `preload=True` nạp sẵn input nhỏ (pooled CT-FM) nếu ≤ 25% RAM trống.

Chạy: `python run.py run data.dataset.<profile> --allow-full` (hoặc `--max-cases N`), preflight bằng `python run.py preflight data.dataset.<profile>`, wrapper `PROFILE=smoke_30 bash scripts/data/preprocessing.sh` (idempotent; biến `PROFILE, ACTION, MAX_CASES, OVERWRITE, GPUS`). Các stage sau chọn profile bằng `--set data.profile=<name>`; run id có hậu tố `__ds_<profile>` khi profile khác `full_inspect`. Stage 0 chỉ cho INSPECT; Turkey đi qua manifest ngoài.

---

## 2. Segmentation pseudo-anatomy

**Code:** `source/segmentation/pipeline.py` (`generate_pseudo_anatomy`), `totalsegmentator.py`,
`lungmask.py`, `resume.py`, `tools/create_masks/generate_masks.py`, `refresh_previews.py`,
`configs/runs/00_data/segmentation/{inspect,turkey}.yaml`, `scripts/data/segmentation.sh`.

Stage này **không train model segmentation**: chạy TotalSegmentator công khai để tạo mask
"pseudo-anatomy"; LungMask chỉ dùng QC chéo mask phổi. Đây là mask của model, không phải annotation
chuyên gia. Input `manifests/ctpa.csv`; output được ROI builder đọc.

```text
manifests/ctpa.csv
  -> NIfTI gốc <raw_inspect>/CT/full/CTPA/<study_id>.nii.gz   (không dùng cache .npy)
  -> TotalSegmentatorRunner: 5 task -> 25 mask project (union / fallback / distance transform)
  -> LungMaskRunner (R231) -> lung_lungmask.nii.gz
  -> mask_qc từng mask + Dice phổi chéo model + _assign_duplicates -> preview
  -> state/<p>/<s>.json -> manifest.parquet, qc_summary.csv, result.json
```

Mask luôn ở **geometry CT gốc** (manifest dataset trỏ cache `.npy` đã resample, còn
TotalSegmentator cần NIfTI gốc, nên `generate_masks.py` đặt `segmentation.raw_image_root`); việc
đưa mask lên grid input làm ở runtime (§1.8). Một worker mỗi GPU (`--gpus`);
`segmentation.workers` tăng số worker (cách duy nhất chạy song song trên CPU).

### 2.1 Task và 26 mask

TotalSegmentator chạy 5 task: `total` (`--roi_subset`: 5 thuỳ phổi, heart), `trunk_cavities`
(mediastinum), `heartchambers_highres` (myocardium, 4 buồng, pulmonary_artery), `lung_vessels`
(airways, arteries, veins), `body` (trunk, extremities). Danh sách 26 mask: `ANATOMIES` trong
`pipeline.py`. Các mask quan trọng cho ROI:

| Mask | Cách dựng |
|---|---|
| `lung`, `lung_left/right`, 5 mask thuỳ | union thuỳ (cùng lần chạy `total`, không tốn thêm inference) |
| `heart` | `total` |
| `strict_heart` | union myocardium + 4 buồng (không gồm PA/aorta); thiếu bộ high-res thì fallback `heart`, ghi `fallback` + `is_approximation` |
| `myocardium, rv, lv, ra, la` | `heartchambers_highres` |
| `central_pa` | class `pulmonary_artery` (gần đúng, không phải ground truth) |
| `mediastinum` | `trunk_cavities` |
| `lung_arteries, lung_veins, airways` | `lung_vessels` |
| `pa_tree` | union `central_pa` + `lung_arteries` |
| `body`, `body_wall` | union trunk + extremities; `body_wall` = voxel body cách bề mặt ≤ 15 mm |
| `hilar_vessels` | `central_pa` ∪ (`lung_vessels` trong vòng 12 mm của mediastinum/central_pa) |
| `lung_lungmask` | LungMask R231, chỉ để QC chéo |

Thuỳ/bên phổi được giữ vì impression thường khu trú PE theo bên và thuỳ. Không có mask phân thuỳ
hay động mạch segmental/subsegmental. `body_wall`/`hilar_vessels` dùng distance transform theo
slab trục z (`source/imaging/morphology.py`) thay cho dilation khối cầu vốn OOM trên VM 16 GB.
Mọi mask được binarize, kiểm tra `same_geometry` với CT; provenance từng mask (`source_task`,
`source_classes`, `postprocessing`, `is_approximation`, `fallback`) đi vào manifest.

### 2.2 Vận hành và weight

- Mặc định `totalseg_resample_threads`/`saving_threads` = 1 vì mặc định 6 tiến trình lưu bị
  OOM-kill trên VM 16 GB. `task_timeout_sec: 3600`: quá hạn thì kill cây tiến trình, mask liên
  quan thành `UNAVAILABLE` thay vì treo cả run. Output thô ghi vào scratch tạm rồi xoá.

### 2.3 LungMask và QC

LungMask CLI dùng `R231`, checkpoint `third_party/weights/segmentation/lungmask/R231.pth` phải có
sẵn (không tự tải). Lỗi LungMask không làm hỏng study: `lung_lungmask` thành `UNAVAILABLE`. Tắt
bằng `segmentation.lungmask.enabled=false` / `LUNGMASK=0`: mọi mask downstream vẫn từ
TotalSegmentator, chỉ mất `lung_lungmask` và bước Dice.

`mask_qc` (`source/imaging/nifti.py`) gán `PASS` / `SUSPICIOUS` (ngoài `qc_volume_ml`, mặc định không đặt) / `FAIL` (không đọc được, lệch geometry, không nhị phân, rỗng) / `UNAVAILABLE` (không tạo được mask). Dice `lung` vs `lung_lungmask`: ≥ 0.90 PASS, < 0.70 FAIL, ở giữa SUSPICIOUS; nếu không PASS thì
status PASS của hai mask bị hạ xuống và `reason` thêm `cross_model_lung_dice_<status>`.
`_assign_duplicates`: hai mask không rỗng cùng SHA-1 voxel → `duplicate_of`.

### 2.4 Preview, resume, output

- Resume: mỗi study xong ghi `state/<patient>/<study>.json`; chạy lại thì tái dùng nếu schema,
  fingerprint settings khớp, đủ 26 anatomy, file mask còn và không có mask chính `UNAVAILABLE`
  (study đó được thử lại). `resume_settings()` loại các key chỉ ảnh hưởng tốc độ / đường dẫn /
  preview (`RESUME_IGNORED_KEYS`). Setting ảnh hưởng output khác đi (ví dụ `lungmask.enabled`) →
  `ResumeSettingsMismatch`; xử lý bằng `--overwrite`, trả lại setting cũ, hoặc đổi experiment id.

Run dir `<output_root>/segmentation/SEG_pseudo_anatomy[__ds_<profile>]/` (Turkey:
`SEG_turkey_pseudo_anatomy`):

```text
masks/<patient_id>/<study_id>/<anatomy>.nii.gz   26 mask/study
previews/  qc_summary.csv  manifest.parquet  state/  resume_settings.json  logs/run.log  result.json
```

`qc_summary.csv` (1 dòng/(study, anatomy), mở đầu tiên): `status, voxel_count, volume_ml,
n_components, duplicate_of, cross_model_dice, reason, mask_path, ...`. `manifest.parquet` là chỉ
mục đầy đủ cho ROI builder (thêm `image_path` NIfTI gốc, provenance, `mask_sha1`, geometry).
`result.json`: có study lỗi hoặc mask chính `FAIL`/`UNAVAILABLE` → `completed_with_failures` và
mã thoát 1; ROI preflight chỉ nhận segmentation run có status `completed`.


Chạy: `PROFILE=smoke_30 GPUS=0 bash scripts/data/segmentation.sh` hoặc `python run.py run data.segmentation --gpus 0 --max-cases 1 --set data.profile=smoke_30` (biến `PROFILE, ACTION, GPUS, MAX_CASES, OVERWRITE, LUNGMASK`). Kiểm tra nhanh: `result.json` status `completed`; lọc `qc_summary.csv` theo `status != PASS`, `voxel_count = 0`, `duplicate_of`; mở vài preview.

---

## 3. ROI1-ROI8 và control ROI

**Code:** `source/roi/builder.py` (`build_roi_dataset`), `random_controls.py`
(`matched_random_control`), `registry.py` (`ROI_DEFINITIONS`), `masks.py`, `counterfactual.py`,
`tools/build_rois/build_rois.py`, `configs/runs/00_data/roi/{inspect,turkey}.yaml`,
`scripts/data/roi.sh`.

Chỉ đọc một segmentation run đã xong (không chạy lại TotalSegmentator), chạy CPU
(`roi.workers`). Mask nguồn dùng được (`_source_reason`): có trong manifest, `status ∈ {PASS,
SUSPICIOUS}`, file tồn tại, cùng geometry CT. Nếu không, ROI ghi `status=failed`,
`qc_severity=UNAVAILABLE`, `roi_path` rỗng kèm `failure_reason`; không bao giờ ghi mask rỗng như
thành công. Body: mask `body` nếu dùng được, nếu không `body_mask_from_hu` (HU > -900, component
lớn nhất, fill holes), nguồn ghi ở `body_source`.

### 3.1 Định nghĩa

| Mã | Tên (`registry.py`) | Operation | Dựng từ |
|---|---|---|---|
| ROI1 | `heart_mediastinum` | KEEP_ONLY | `heart` ∪ `mediastinum` |
| ROI2 | `strict_heart` | KEEP_ONLY | mask `strict_heart` (có thể là fallback `heart`, xem `source_fallbacks`) |
| ROI3 | `dilated_central_pulmonary_artery` | REMOVE_ROI | `central_pa` giãn 2.0 mm (theo không gian vật lý) |
| ROI4 | `pulmonary_artery_tree` | KEEP_ONLY | `pa_tree` |
| ROI5 | `whole_lung_with_vessels` | KEEP_ONLY | `lung` |
| ROI6 | `lung_parenchyma_without_large_vessels` | KEEP_ONLY | `lung` trừ `lung_arteries`, `lung_veins` (trừ `airways` chỉ khi `roi.subtract_airways: true`, mặc định false) |
| ROI7 | `heart_hilar_vessels_exclusion` | REMOVE_ROI | `heart` ∪ `hilar_vessels` |
| ROI8 | `control_for_<ROI>_<tên>` | KEEP_ONLY | dịch chuyển cứng ROI2/ROI4/ROI6 (§3.2) |

ROI3/ROI7 là vùng **xoá**, còn lại là vùng **giữ**. Mỗi dòng có `recipe` (JSON) ghi công thức
(`x*M + replacement*(1-M)` cho keep, `x*(1-M) + replacement*M` cho remove) và `masking_policy`
(mặc định `local_mean`, bán kính 3 voxel). QC mỗi ROI: `mask_qc` (rỗng → FAIL); `qc_severity` =
`FAIL` / `SUSPICIOUS` (QC hoặc mask nguồn SUSPICIOUS) / `PASS`; voxel ngoài body → `outside_body`,
`status=failed`. `status` là `pass` hoặc `failed`.

### 3.2 ROI8: control khớp thể tích

Mục đích: một vùng "giả" cùng hình dạng và đúng số voxel như ROI nguồn nhưng ở chỗ khác, vì thay
đổi dự đoán sau khi xoá một vùng chỉ có ý nghĩa khi so với xoá một vùng cùng kích thước bất kỳ.

```text
target    = ROI nguồn (ROI2 | ROI4 | ROI6)
region    = (body ∪ body_wall) ∩ isfinite(CT)
forbidden = heart ∪ mediastinum ∪ central_pa ∪ lung_arteries ∪ lung_veins ∪ hilar_vessels
candidate = region ∩ ¬dilate(target ∪ forbidden, 5 mm)
offset    : dịch cứng target, preserve_z_range = chỉ dịch trong mặt phẳng axial (trục trên-dưới lấy từ affine)
rút tối đa 500 offset (seed = SHA-256 của "<seed>|<patient>|<study>|<ROI>", seed 42)
nhận offset đầu tiên mà mọi voxel dịch đều nằm trong candidate; không có -> no_valid_control_location
```

Chỉ dịch chuyển cứng (không co, crop, resample) nên `actual_voxels == target_voxels`;
`require_exact_voxel_match: true` kiểm tra lại. Mọi mask trong `excluded_anatomies` phải dùng
được, nếu không ROI8 failed với `forbidden_anatomy_unavailable:...`. Cột ROI8 thêm gồm
`seed, method, target_voxels, actual_voxels, forbidden_overlap_voxels, dice_with_source,
translation_voxels, attempts_tested, ...` (danh sách đủ trong `random_controls.py`).

> **ROI8 gần như luôn thất bại cho ROI4 và ROI6.** Với `preserve_z_range: true`, control chỉ dịch
> in-plane. Vùng loại trừ gồm chính ROI nguồn (giãn 5 mm) cộng toàn bộ mạch phổi, tim, trung thất.
> ROI6 (nhu mô hai phổi) và ROI4 (cây PA trải từ trung thất ra hai phổi) chiếm gần hết mặt cắt
> ngực ở các lát của chúng, nên không có độ dịch in-plane nào vừa thoát chính nó + vùng cấm vừa
> còn trong body. Kết quả mong đợi: dòng ROI8 `failed` với `failure_reason=no_valid_control_location`.
> ROI2 (nhỏ hơn) có cơ hội cao hơn nhưng cũng có thể thất bại. **Luôn kiểm tra `roi_manifest.csv`
> thay vì giả định ROI8 có sẵn.** Hệ quả: `anatomy.remove_random`
> (`roi_control_for: {random: ROI4}`) cần ROI8 cho ROI4; study không có control không nạp được
> (`CTPADataset` raise `required ROI mask is unavailable`). Đổi `preserve_z_range`,
> `exclusion_margin_mm` hay `excluded_anatomies` là đổi protocol (resume sẽ chặn).

### 3.3 Output, resume, lệnh chạy

Run dir `<output_root>/roi/ROI_anatomy_and_controls[__ds_<profile>]/` (Turkey:
`ROI_turkey_anatomy_and_controls`):

- `roi_manifest.csv`: một dòng mỗi ROI1-ROI7 và mỗi cặp (ROI8, nguồn). Cột chính: `patient_id,
  study_id, split, roi_id, control_for, operation, status, qc_severity, failure_reason, roi_path,
  source_anatomies, source_fallbacks, recipe, voxel_count, volume_ml, geometry_match,
  body_contained` + cột ROI8. Đường dẫn tuyệt đối.
- `result.json`: `completed_with_failures` (mã 1) chỉ khi một study lỗi ngoại lệ; ROI `failed` đơn
  lẻ (ví dụ ROI8) không làm run thất bại.
- Resume: tái dùng state khi fingerprint settings khớp, `source_digest` (hash `(anatomy, status,
  mask_sha1)` của segmentation study đó) khớp, file ROI còn. Segmentation dựng lại một study →
  chỉ study đó tính lại. Setting khác → `ResumeSettingsMismatch`; dùng `--overwrite`.

### 3.4 ROI được dùng thế nào sau đó

`configs/components/anatomy.yaml#anatomy_masks` đặt `data.roi_manifest:
roi/ROI_anatomy_and_controls/roi_manifest.csv` và `data.roi_mask_ids: {heart: ROI2, pa: ROI4,
lung: ROI6}`; arm counterfactual random thêm `{random: ROI8}` với `roi_control_for: {random: ROI4}`.
`MaskingPolicy`: `zero`, `global_mean`, `local_mean` (mặc định), `noise_matched`. Region `random`
bắt buộc là ROI8 tính sẵn: `matched_random_mask` luôn raise (cấm tạo control lúc runtime). Đường dẫn
trên là run của `full_inspect`; train anatomy trên profile khác phải trỏ `data.roi_manifest` (và
`supervision.masks`/`supervision.rois`) tới run `__ds_<profile>` bằng `--set`.

---

## 4. Silver labels

**Code:** `source/silver/` (`schema.py`, `providers.py`, `extractor.py`, `generator.py`,
`rules.py`, `qc.py`, `audit.py`), `tools/silver_labels/generate_silver_labels.py`,
`configs/components/silver.yaml`, `configs/runs/00_data/silver/medgemma.yaml`,
`scripts/data/silver_labels.sh`.

INSPECT chỉ có vài nhãn native (`pe_positive_nlp`, `pe_acute`, `pe_subsegmentalonly`). Các thuộc
tính giàu hơn (vị trí cục máu đông, acuity, RV, tràn dịch, khí phế thũng...) chỉ có trong text
report. Stage này trích **19 target** thành nhãn "silver" bằng MedGemma + regex.

- **Chỉ là auxiliary supervision**: loss phụ cho các head phụ khi train diagnosis. Không bao giờ
  dùng làm nhãn đánh giá (gold).
- **Ưu tiên abstain hơn đoán**: một dòng accepted sai tệ hơn một dòng thiếu. Dòng không accepted
  có `value` rỗng và bị mask khỏi loss, không bao giờ thành nhãn âm.
- **Chỉ có text**: MedGemma không nhận CT. Report INSPECT chủ yếu là IMPRESSION; thứ impression
  không nhắc (RV, septal bowing...) sẽ bị abstain, đó không phải lỗi parse.

### 4.1 Target schema (`source/silver/schema.py`)

`TARGETS` là tuple 19 `TargetSpec(name, kind, values, aliases, ...)`; danh sách đầy đủ và mô tả
nằm ở đó. Nhóm chính (branch organ = cơ quan nhận loss phụ trong `diagnosis_organ_silver`):

| Nhóm | Target | Kind | Branch |
|---|---|---|---|
| PE | `pe_present` | binary | (native) |
| Acuity | `acuity` (`acute, chronic, acute_on_chronic, uncertain`) | categorical | pa |
| Vị trí | `central, lobar, segmental, subsegmental, saddle` | binary | pa |
| RV / tim | `rv_enlargement, rv_lv_ratio_abnormal` (alias `rv_lv_abnormal`), `septal_bowing`, `contrast_reflux` (alias `reflux`), `pericardial_effusion` | binary | heart |
| RV ratio | `rv_lv_ratio_mentioned`, `rv_lv_ratio_value` (continuous, ≥ 0) | binary / continuous | không gắn branch |
| Phổi | `pleural_effusion, chronic_lung_disease, fibrosis, emphysema` | binary | lung |
| Khác | `malignancy_related_finding` | binary | không gắn branch |


### 4.2 MedGemma extractor

- Provider `TransformersProvider` (`providers.py`): checkpoint local `third_party/weights/medgemma`
  (`google/medgemma-1.5-4b-it`), `AutoModelForImageTextToText` (checkpoint là
  `Gemma3ForConditionalGeneration`), bf16, `local_files_only`, greedy, `max_new_tokens: 384` (192
  từng cắt JSON giữa chừng). Chat template gọi tokenizer với `add_special_tokens=False` để tránh
  BOS kép.
- Prompt `v2`: luật `true` khi report ghi có (mọi kích thước/bên), `false` khi ghi không có hoặc có
  câu loại trừ rõ ràng ("No pulmonary embolism" → `pe_present` và cả 5 vị trí false), `null` khi không nhắc
  hoặc nước đôi; trả một object JSON `target, value, confidence, reason, evidence_text`.
- Parse mềm (`parse_json_response`): sửa lỗi format vô hại (`"yes"` → true, `"subacute"` → acute,
  `"95%"` → 0.95, chuỗi số → số), mọi chuyển đổi ghi vào `normalization`; thiếu `value` hoặc sai
  kiểu sau chuẩn hoá → `ProviderResponseError`. `evidence_text` luôn được tìm lại trong report để
  tính offset; không thấy → `evidence_valid=False`. Confidence là model **tự báo**, không phải xác
  suất đã calibrate.
- Retry: `retries: 1` → tối đa 2 lần, lần hai gửi kèm thông báo lỗi trước đó; hết lần → `ProviderResponseError`.

### 4.3 Generator: quyết định từng target (`generator.py`)

MedGemma được coi là **confident** khi `value` khác null, evidence không bị bác, và `confidence ≥
0.8` (`medgemma_confidence_threshold`, cố định, chưa calibrate trên validation, không tune trên test).

```text
MedGemma lỗi:   rule resolved -> ACCEPT (rule) | không -> NO_RESULT  technical_failure:...
MedGemma confident:
   rule resolved và khác giá trị -> ABSTAIN  medgemma_rule_conflict
   còn lại                       -> ACCEPT   medgemma_confident[_rule_agrees]
MedGemma không confident:
   rule resolved và (MedGemma null hoặc cùng giá trị) -> ACCEPT (rule)  rule_..._after_medgemma_abstained
   rule resolved nhưng MedGemma ngược                 -> ABSTAIN  medgemma_rule_conflict_unconfident
   evidence không có trong report                     -> ABSTAIN  medgemma_evidence_not_in_report
   null / dưới ngưỡng                                 -> ABSTAIN  medgemma_null_value / medgemma_below_confidence_threshold
```

Với 5 target vị trí, nếu regex không thấy gì mà `pe_present` resolve False thì coi là
`implied_by_explicit_no_pe` (rất hay gặp ở report "No PE"). Exception ngoài provider → `no_result`
(`source=pipeline`); report rỗng → 19 dòng `no_result` `empty_report`.

**`pe_consistency` (`_enforce_pe_consistency`)** chạy sau khi đủ 19 target, dùng `pe_present` đã
accepted. Nếu `pe_present = False`: vị trí accepted `true` hoặc MedGemma nói `true` → **abstain**
(không che mâu thuẫn), các vị trí còn lại được accept `false` (`implied_by_pe_present_false`),
`acuity` accepted → abstain. Nếu `pe_present` chưa quyết: vị trí `false` và `acuity` accepted đều
abstain (false khi chưa biết có PE là đoán). Các reason `pe_consistency:*` khác nằm trong code.

### 4.4 Regex rules (`rules.py`, `RULE_VERSION = "pe_rules_v4"`)

Regex là backup giải thích được, không phải stage riêng. Nguyên tắc: **chỉ resolve khi mọi mention
trong phần finding đồng ý**; nghi ngờ thì unresolved. `apply_rule(report, target)`:

1. Che section không phải finding (INDICATION, HISTORY, COMPARISON, TECHNIQUE...).
2. Phạm vi cue chỉ trong mệnh đề của nó (ngắt bởi dấu chấm không theo số, `;`, xuống dòng, liên từ
   đối lập `but/however/except`).
3. Phân loại mention theo thứ tự: `ignored` (bệnh sử `history of`, chỉ định "Eval for PE", nghi ngờ
   trong preamble) → `uncertain` (`cannot exclude`, `possible`, `likely`, `versus`, `?`) →
   `affirmed` (dạng "Finding: yes") → `negated` (cue phủ định trong ≤ 8 từ, hoặc phủ định đặt sau;
   pseudo-negation như "no change in" không phủ định; "partially resolved" không phải phủ định) → `affirmed`.
4. Tổng hợp: có `uncertain` → unresolved; vừa affirmed vừa negated → `conflicting_explicit`; chỉ
   affirmed → true; chỉ negated → false; không có gì → `no_explicit_evidence`.


### 4.5 Audit, QC, resume

- `silver_qc_summary()` → `result.json` mục `evaluation`: `coverage`, `abstention_rate`,
  `no_result_rate`, `by_target`, `reason_counts`, `issue_counts` (`conflict`/`low_conf`/
  `missing_field`/`impossible_value`), `missingness_by_target`, `rule_medgemma_disagreement`.
  `expert_review_queue` luôn `false`.
- **Cache `.state/<report_hash>.json`** dùng lại khi: schema version, chữ ký generator
  (`prompt_version`, `RULE_VERSION`, `rule_rescue`, `pe_consistency`, threshold, `model_id`,
  `retries`, checkpoint files, `max_new_tokens`, `prompt_sha256`), `input_signature` của report và
  đủ 19 rows đều khớp, và **không có lỗi kỹ thuật** (`no_result`, `technical_failure`,
  `after_medgemma_failure`). Sửa prompt template / description / allowed values đều làm mất cache;
  report từng lỗi luôn được hỏi lại ở lần sau. Resume là mặc định, `--overwrite` xoá run cũ.

### 4.6 Output và cách dùng downstream

Output `<output_root>/silver_label/medgemma/`: `silver_labels.csv` (một dòng/report × target:
`patient_id, study_id, report_id, split, target, value` (JSON scalar), `status, source`
(`medgemma|rule|pipeline`), `confidence`, `reason`, `checkpoint_id`, ...),
`silver_label_confidence.csv` (thêm `decision, abstain_reason, evidence_text`; không giả confidence
thành xác suất), `result.json`, `logs/`, `.state/`.

Downstream:
1. Config diagnosis đặt `supervision.require_silver_labels: true` và `supervision.silver_labels:
   silver_label/medgemma/silver_labels.csv` (tương đối với output root).
2. Preflight (`audit_silver_table`): cột bắt buộc, status hợp lệ, accepted có value; mỗi
   `task.silver_targets` phải có ít nhất một dòng accepted; không hai dòng accepted cho cùng
   (patient, study, target).
3. `CTPADataset` chỉ đọc `accepted`: binary → 0/1, categorical → index trong `values`
   (acuity: `acute`=0, `chronic`=1, `acute_on_chronic`=2, `uncertain`=3), continuous → số. Không có
   dòng accepted → NaN, `silver_valid=False` (bị mask).
4. Loss (`source/engine/task_steps.py::task_loss_step`): `diag.global.silver_multitask` cộng
   `silver_loss_weight` (0.2) × loss silver lên các head cùng tên; `diag.anatomy.silver.
   {concat,late,moe}` gắn target vào branch heart/pa/lung qua `auxiliary_targets`
   (`organ_auxiliary_loss`, trọng số 0.2 mỗi organ, `silver_loss_weight: 0.0` để không cộng hai
   lần). `source/tasks/diagnosis/organ_targets.py` từ chối target gán sai branch hoặc cho hai branch.
5. **Evaluation không bao giờ dùng silver** (`diagnosis_evaluation_targets` bỏ target trong `silver_targets`).
