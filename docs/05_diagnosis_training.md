# 05. Quy trình training diagnosis

## Mục tiêu

Dự đoán PE presence và các target phụ từ CTPA. Các baseline image-only là mốc so sánh bắt buộc.

## Các nhánh

- Global single: toàn volume, target `pe_present`.
- Global native multitask: PE và native attributes.
- Anatomy: image encoder + nhánh heart/PA/lung, fusion concat/late-logit/Soft-MoE.
- Anatomy + silver: thêm accepted SL02 organ targets.
- Report-only: baseline từ report.

## Luồng chạy

1. Preflight manifest và checkpoint C0.
2. Load `diagnosis.csv` theo train/validation/test split.
3. Load image encoder từ C0.
4. Nếu anatomy: đọc pseudo masks/ROI theo manifest segmentation.
5. Nếu silver: đọc `labels.parquet`, mask các dòng không accepted.
6. Train với native diagnosis target; validation/test chỉ dùng label thật.
7. Ghi checkpoint, metrics, lineage và report.

## Ví dụ

```bash
python run.py preflight diag.global.single
python run.py run diag.global.single --gpus 0
python run.py run diag.anatomy.silver.concat --gpus 0
```

## Checklist lỗi

- Baseline global không được vô tình đọc mask hoặc EHR.
- Silver label không được biến thành native target.
- Train/validation/test phải patient-level và không leakage.
- So sánh concat, late-logit, Soft-MoE chỉ khi backbone, split và hyperparameter giống nhau.
- Kiểm tra `task.targets`, `label_columns` và số output head khớp nhau.

