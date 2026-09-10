# 08. Tổng hợp các bước chạy

## Thứ tự chuẩn

```text
01 Dataset build
   ↓
02 Segmentation pseudo-masks
   ↓
03 ROI / anatomy artifacts (nếu cần)
   ↓
04 Silver labels (nếu dùng silver supervision)
   ↓
05 DAPT / image-report alignment → C0
   ↓
06 Diagnosis baselines → anatomy/silver diagnosis
   ↓
07 Prognosis baselines → multimodal/anatomy prognosis
   ↓
08 Contour (deferred, chỉ khi có reviewed embolus masks)
```

## Trước mỗi stage

```bash
python run.py preflight <experiment>
python run.py plan <experiment>
```

Kiểm tra: manifest tồn tại, split không leakage, checkpoint đúng lineage, weights/model có thật và output root writable.

## Các blocker hiện tại cần xử lý

1. Backbone configs còn `factory/output_adapter` rỗng và `feature_dim: 0`.
2. Alignment chưa có `report_embedding_columns` thật.
3. EHR columns chưa cấu hình đủ 32 chiều.
4. PESI còn chờ clinical approval.
5. RSPECT mới có ZIP tại `/mnt/RSPECT_dataset`; cần giải nén và tạo patient-level manifest.
6. Contour còn trỏ checkpoint `DX01` cũ.

## Quy tắc debug

- Chạy smoke/small cohort trước full.
- Mỗi stage phải đọc `result.json`, `data_quality`, `qc.json` hoặc manifest output.
- Không bỏ qua `FAIL`, `SUSPICIOUS`, `UNAVAILABLE` bằng cách đổi status thủ công.
- Khi sửa bug, ghi rõ input, expected output và test/preflight đã chạy.

