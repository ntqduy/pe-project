# 01. Quy trình xử lý dữ liệu

## Mục tiêu

Tạo một dataset profile thống nhất cho mọi stage downstream. Code chính: `source/data_preprocessing/pipeline.py`.

## Luồng xử lý

1. Đọc dữ liệu INSPECT read-only từ `raw_inspect`.
2. Join study/patient/split và giữ official split.
3. Loại patient trong governance exclusion registry.
4. Lọc eligibility: thiếu file, acquisition không hợp lệ.
5. Kiểm tra CT integrity: file đọc được, số lát tối thiểu, QC volume.
6. Adjudicate label ở cấp patient.
7. Sampling theo patient nếu profile yêu cầu.
8. Kiểm tra leakage và giữ lại toàn bộ study của patient được chọn.
9. Nếu bật preprocessing: chuẩn hóa CT và ghi cache volume.
10. Tạo manifest, clinical artifacts, audit và provenance.

## Input/output

- Input: `${PE_RAW_INSPECT_ROOT}/CT/{full,sample}` hoặc `paths.raw_inspect`.
- Output: `${PE_DERIVED_ROOT}/datasets/<profile>/`.
- File quan trọng: `manifests/ctpa.csv`, `diagnosis.csv`, `prognosis.csv`, `paired_reports.csv`, `clinical/*`, `audit/*`, `data_quality.md/json`, `dataset.json`.

## Chạy

Hai cách gọi tương đương. Wrapper nhận biến môi trường, `run.py` nhận cờ CLI:

```bash
# wrapper
bash scripts/0_data_preprocessing/build_smoke_30.sh
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_test_500_sample.sh
ALLOW_ALL=1 bash scripts/0_data_preprocessing/build_full_inspect.sh

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

