# 01. Quy trình xử lý dữ liệu

## Mục tiêu

Tạo một dataset profile thống nhất cho mọi stage downstream. Code chính: `source/data_preprocessing/pipeline.py`.

## Luồng xử lý

1. Đọc dữ liệu INSPECT read-only từ `raw_inspect`.
2. Join study/patient/split và giữ official split.
3. Nếu profile bật `sampling.before_eligibility`, lấy mẫu patient theo seed từ metadata/nhãn ứng viên.
4. Loại patient trong governance exclusion registry.
5. Lọc eligibility: thiếu file, acquisition không hợp lệ.
6. Kiểm tra CT integrity: file đọc được, số lát tối thiểu, QC volume.
7. Adjudicate label ở cấp patient; profile lấy mẫu sau eligibility sẽ sampling tại đây.
8. Kiểm tra leakage và giữ lại toàn bộ study của patient được chọn.
9. Nếu bật preprocessing: chuẩn hóa CT và ghi cache volume.
10. Tạo manifest, clinical artifacts, audit và provenance.

Riêng `smoke_30` lấy 30 candidate (10 mỗi official split) trước eligibility và integrity để
tránh quét CT toàn cohort; vì vậy dataset cuối có thể ít hơn 30 patient. `test_500_sample`
vẫn lấy 500 patient sau eligibility để giữ định nghĩa cohort rehearsal hiện tại.

## Input/output

- Input: `${PE_RAW_INSPECT_ROOT}/CT/{full,sample}` hoặc `paths.raw_inspect`.
- Output: `${PE_DERIVED_ROOT}/datasets/<profile>/` (cache CT/EHR: `${PE_DERIVED_ROOT}/cache/<profile>/`).
- File quan trọng: `manifests/*.csv` (ID, split, nhãn theo task), `manifests/exclusions.csv` (ID bị loại), `data_quality.md` (số missing theo task), `dataset.json` và `logs.txt`.

## Chạy

Hai cách gọi tương đương. Wrapper nhận biến môi trường, `run.py` nhận cờ CLI:

```bash
# wrapper
python run.py run data.dataset.smoke_30
python run.py run data.dataset.test_500_sample --allow-full
python run.py run data.dataset.full_inspect --allow-full

# run.py trực tiếp
python run.py preflight data.dataset.smoke_30
python run.py run data.dataset.smoke_30 --max-cases 30
python run.py run data.dataset.test_500_sample --allow-full
```

Biến của wrapper: `MAX_CASES`, `PATIENT_ID`, `ALLOW_ALL`, `ACTION`
(`show|plan|preflight|dry|run`), `SET` cho override thô. Xem `scripts/README.md`.

## Checklist lỗi

- Không chạy full nếu chưa dùng `--allow-full`.
- Sampling phải ở cấp patient, không ở cấp study.
- Mọi manifest phải giữ patient/study/split rõ ràng.
- Kiểm tra `dataset.json` và `data_quality.md` trước khi train.
- Dataset build hiện phụ thuộc INSPECT; không dùng root RSPECT cho stage này.

