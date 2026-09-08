# Pipeline build dataset INSPECT

Tài liệu này mô tả chính xác luồng mã được gọi bởi ba dataset wrapper:

```bash
bash scripts/0_data_preprocessing/build_full_inspect.sh
bash scripts/0_data_preprocessing/build_test_500_sample.sh
bash scripts/0_data_preprocessing/build_smoke_30.sh
```

Ba profile dùng **cùng một implementation** cho việc đọc dữ liệu, lọc, kiểm tra CT,
adjudicate nhãn, tạo manifest và tiền xử lý volume. Chúng chỉ khác ở bước sampling.
Raw INSPECT release chỉ được đọc, không bị sửa.

## Lưu ý: hai lệnh nguyên văn không build gì

`scripts/_lib.sh` yêu cầu một scope rõ ràng cho data-generation stage. Vì vậy, hai
lệnh ở trên mặc định (`ACTION=run`) dừng ngay trong `pe_run` với exit code 2 và gợi ý
dùng `MAX_CASES`, `PATIENT_ID` hoặc `ALLOW_ALL`; `run.py` còn chưa được gọi và chưa có
file cohort, cache hay manifest nào được tạo.

Nếu `DATASET` không được đặt, mỗi dataset wrapper tự chọn đúng profile của nó. Các
training stage khác vẫn mặc định dùng `test_500_sample`.

Dùng các lệnh sau để build theo thứ tự kỹ thuật an toàn:

```bash
# Smoke độc lập: 30 bệnh nhân, 10 từ mỗi official split
bash scripts/0_data_preprocessing/build_smoke_30.sh

# Rehearsal: toàn bộ profile 500 bệnh nhân
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_test_500_sample.sh

# Full: toàn bộ cohort INSPECT hợp lệ
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_full_inspect.sh
```

`ACTION=preflight` kiểm tra cấu hình/input nhưng không build; `OVERWRITE=1` mới cho
phép build lại một thư mục output đã có `dataset.json`.

## Chuỗi gọi mã

```text
build_{smoke_30|full_inspect|test_500_sample}.sh
  -> scripts/_lib.sh : pe_run --scoped <experiment>
  -> python run.py run <experiment> --set data.profile=<DATASET> --allow-full
  -> tools/data/build_dataset.py --config configs/runs/00_data/dataset/<profile>.yaml
  -> source.data_preprocessing.pipeline.build_dataset(profile, paths, ...)
```

`run.py` tra experiment trong `configs/experiments.yaml`, resolve YAML cấu hình, rồi
delegates dataset stage cho `tools/data/build_dataset.py`. YAML run config bật
`dataset.preprocess: true`; profile contract được load từ
`source/dataset/profiles/{smoke_30,full_inspect,test_500_sample}.yaml`, cùng kế thừa
`_common.yaml`.

## Các bước thực thi trong `build_dataset`

1. **Resolve đường dẫn.** `ProjectPaths` lấy raw release từ `PE_RAW_INSPECT_ROOT`/cấu
   hình; release mặc định là `<raw_inspect>/CT/full`. Output là
   `${PE_DERIVED_ROOT}/datasets/<profile>/`. Trong workspace này, bạn tự mount
   `gs://pe-study/pe-storage` vào `/mnt/pe-storage`; sau đó
   `source scripts/use_gcs_storage.sh` đặt nó thành
   `/mnt/pe-storage/derived/datasets/<profile>/`. Helper này không tự mount.

2. **Đọc và join release INSPECT.** `InspectSource` tự tìm các TSV date-stamped:
   `splits`, `labels`, `study_mapping`, `series_metadata`, `study_metadata`,
   `impressions` (và EHR crosswalk nếu có). Nó join thành `StudyRecord`; đổi split
   `valid` thành `validation`; chọn series có nhiều slices nhất (tie-break: slice
   mỏng hơn); và lấy giá trị metadata phổ biến nhất cho mỗi study.

3. **Áp dụng governance và eligibility.** Nếu có registry loại bệnh nhân, loại toàn
   bộ bệnh nhân đó trước. Mỗi study còn lại phải có split hợp lệ, NIfTI CT tồn tại,
   report, labels và series metadata; modality phải là CT; `num_slices >= 2`,
   `SliceThickness <= 5.0 mm`, `PixelSpacing_0/1 <= 2.0 mm`. Một study chỉ ghi nhận
   rule thất bại đầu tiên vào ledger `audit/exclusions.csv`.

4. **Kiểm tra integrity CT.** Mặc định `level: header`: file phải đủ lớn, NIfTI header
   parse được, có hình học 3-D hợp lý và ít nhất 2 slices. CT missing/corrupt bị loại
   khỏi cohort và được ghi vào `audit/integrity.json` cùng `audit/exclusions.csv`. (Không phải
   `level: volume`, nên voxel không được decode toàn bộ ở bước này.)

5. **Adjudicate nhãn theo bệnh nhân.** Bốn nhãn `pe_positive_nlp`, `pe_acute`,
   `pe_subsegmentalonly`, `1_month_mortality` được gộp từ các studies của cùng bệnh
   nhân theo thứ tự `TRUE > CENSORED > FALSE > MISSING`. Mã dừng lỗi nếu một bệnh nhân
   đã xuất hiện ở nhiều official split.

6. **Chọn cohort profile.**

   | Profile | Cách chọn sau bước lọc/QC |
   | --- | --- |
   | `smoke_30` | 30 bệnh nhân deterministic: 10 từ mỗi official split. Chỉ dùng để kiểm tra kỹ thuật, không dùng để chọn hay báo cáo model. |
   | `full_inspect` | Giữ toàn bộ study hợp lệ; không sampling. |
   | `test_500_sample` | Seed `20260704`; lấy đúng 500 **bệnh nhân** theo tỷ lệ số bệnh nhân của train/validation/test, stratify theo 4 nhãn trên, rồi giữ mọi study của bệnh nhân đã chọn. Prevalence tự nhiên được giữ vì `positive_fraction: null`. |

   Nếu dùng `MAX_CASES=N`, mã giới hạn theo **N bệnh nhân đầu tiên trong cohort** và
   vẫn giữ toàn bộ studies của mỗi bệnh nhân đó.

7. **Guard chống leakage.** Audit xác nhận không có bệnh nhân cross-split, study ID
   trùng, study lạ, hay split bị đổi so với release; build lỗi nếu audit không đạt.
   Full build cũng yêu cầu có cả train, validation và test.

8. **Tiền xử lý CT và tạo cache.** Với từng study trong cohort, load NIfTI, chuẩn hoá
   orientation về RAS khi có thể, clip HU `[-1000, 1000]`, min-max scale về `[0, 1]`,
   center crop/zero-pad thành `128 x 128 x 128`, cast `float32`, rồi ghi
   `volumes/<study_id>.npy`. Không resample spacing. Cache đã có sẽ được tái dùng trừ
   khi `OVERWRITE=1`; lỗi từng case được ghi trong `dataset.json` và case đó không vào
   manifest. Mỗi cache entry cũng được hậu kiểm shape, dtype, finiteness và range trước
   khi được đưa vào manifest; chi tiết nằm trong `audit/cache_qc.json`.

9. **Sinh manifests và provenance.** Manifest trỏ tới cache `.npy` (không trỏ raw NIfTI):

   - `ctpa.csv`: mọi study còn lại, cho segmentation/DAPT.
   - `diagnosis.csv`: mọi study, native labels `pe_present`, `pe_acute`,
     `pe_subsegmental_only`; CENSORED/MISSING được để trống, không biến thành 0.
   - `paired_reports.csv`: mọi cặp image--impression.
   - `reports.csv`: report dùng cho silver-label generation.
   - `prognosis.csv`: một index study sớm nhất cho mỗi bệnh nhân có
     `pe_positive_nlp=TRUE` và `pe_acute=TRUE`; mortality censored/missing vẫn để
     missing thay vì coi là sống.

   Entry point để review là `data_quality.md` (một trang aggregate, không có patient ID)
   và `data_quality.json`. Bảng đầu tiên trong đó là **cohort funnel**: số study và số bệnh
   nhân còn lại sau *từng* bước (`release` → `governance` → `eligibility` → `ct_integrity`
   → `sampling` → `scope` → `preprocessing` → `final`), kèm số bị loại ở mỗi bước, nên không
   cần cộng tay ba report riêng lẻ để biết còn bao nhiêu sample. Chi tiết patient-linked được gom trong `audit/`:
   `exclusions.csv`, `integrity.json`, `split_audit.json`, `cache_qc.json`.
   `dataset.json` lưu full provenance, scope, sampling, preprocessing fingerprint và
   thống kê manifest.

## EHR và PESI

Sau CTPA QC, stage 0 đọc `meds_omop_inspect.tar.gz` vào **local cache** (không copy raw
archive vào từng profile), stream các event của cohort, rồi viết:

```text
clinical/ehr_features.csv          one row / CTPA; chỉ utilization features an toàn
clinical/ehr_train_vocabulary.csv  code vocabulary fit trên official train duy nhất
clinical/ehr_metadata.json         archive provenance, missingness, event exclusions
clinical/pesi_status.json          trạng thái PESI/sPESI có kiểm soát
clinical/pesi_mapping_audit.json   code/source/unit/timing review cho 11 component
# clinical/pesi_features.csv       chỉ có khi mapping được duyệt và score thực sự được tạo
```

EHR chỉ nhận event thỏa `event_time < procedure_datetime - 0h`; policy
`configs/clinical/ehr_prohibited_features.json` loại `note` và code/result proxy của
CTPA/radiology. Các cột EHR readiness được merge vào `ctpa.csv`, `diagnosis.csv` và
`prognosis.csv`; `data_quality.md` báo missing index time, EHR-missing và các event bị
loại. Đây là feature readiness, **không** phải tự suy diễn clinical variables.

PESI/sPESI là một **clinical gate**, không phải một phép join tự động. Pipeline kiểm tra
candidate source codes trong `configs/clinical/pesi_spesi_mapping.yaml` rồi ghi mapping
audit và aggregate availability/unit/time audit (không có patient ID hay raw value). Hiện
mapping ở trạng thái `pending`: codebook có candidate cho birth/sex, heart rate,
respiratory rate, temperature và oxygen saturation, nhưng chưa có systolic-BP đã
xác minh, phenotype được duyệt cho cancer/heart-failure/chronic-lung/altered-mental-status,
hoặc review unit và cửa sổ 6 giờ trước CTPA. Vì vậy default build kết thúc bình thường với
`clinical/pesi_status.json: blocked`, **không** tạo score giả.

Khi clinical steward duyệt đủ 11 component, đổi `clinical_approval.status: approved`, đặt
từng component thành `approved`, và cung cấp một CSV study-level qua
`pesi.components_table` với `study_id` cùng các cột component. Khi đó stage 0 tạo
`clinical/pesi_features.csv`, merge `pesi`, `spesi`, `pesi_class` và các cờ computability
vào manifest. Preflight của mọi prognosis arm dùng PESI chỉ PASS khi artifact có trạng
thái `built` và có cả hai score hữu hạn trong train/validation/test.

Silver label và segmentation là stage riêng; chúng tiêu thụ lần lượt `manifests/reports.csv`
và `manifests/ctpa.csv` do stage này tạo.

## Stage 0 nối vào các stage sau như thế nào

| Artifact stage 0 tạo | Ai đọc | Cách trỏ tới |
| --- | --- | --- |
| `manifests/ctpa.csv` | segmentation, ROI, DAPT, alignment | `data.manifest`, tương đối so với dataset root |
| `manifests/diagnosis.csv` | mọi arm diagnosis | `data.manifest` |
| `manifests/prognosis.csv` | mọi arm prognosis | `data.manifest` |
| `manifests/reports.csv` | silver label | `silver.reports` |
| `volumes/<study_id>.npy` | mọi stage đọc ảnh | cột `image_path` trong manifest |
| `clinical/ehr_features.csv` | prognosis (`ehr`) | `supervision.ehr`, tương đối so với dataset root |
| `clinical/pesi_features.csv` | prognosis (`pesi`) | `supervision.pesi`, chỉ tồn tại khi PESI được duyệt |

**Mask giải phẫu không nằm trong manifest stage 0.** Không có cột `*_mask_path` nào được
sinh ra và cũng không được bịa ra. Heart/PA/lung mask là artifact của stage 1: chúng được
đọc trực tiếp từ ROI run manifest qua `data.roi_manifest` + `data.roi_mask_ids`
(`configs/components/data/anatomy_masks.yaml`), dùng đúng ROI id và đúng bộ lọc QC
(PASS/SUSPICIOUS) mà ROI student và counterfactual đang dùng:

```text
heart -> ROI2 (strict heart)   pa -> ROI4 (PA tree)   lung -> ROI6 (lung parenchyma)
```

Nhờ vậy một branch, vùng bị xoá khỏi nó và student học riêng vùng đó luôn trỏ về cùng một
tập voxel.

## Kết quả và các điều không được chạy

Hai script này chỉ xây dựng dataset. Chúng không chạy TotalSegmentator, ROI, silver
labels, foundation pretraining, diagnosis/prognosis training, evaluation hay
counterfactual analysis. Các stage đó có wrapper/experiment riêng và tiêu thụ các
manifest được tạo ở đây.

Luồng thực tế có thể kiểm tra trước bằng:

```bash
python run.py preflight data.dataset.test_500_sample
python run.py preflight data.dataset.full_inspect
python run.py preflight data.dataset.smoke_30
```
