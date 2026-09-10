# 03. Quy trình tạo silver label

## Mục tiêu

Trích xuất nhãn từ radiology report bằng rule model, Falcon và/hoặc MedGemma. Silver label chỉ dùng làm supervision phụ, không thay expert label khi đánh giá.

## Các method

- `SL00`: MedGemma.
- `SL01`: Falcon.
- `SL02`: Falcon + MedGemma + adjudication; bất đồng thì abstain.

## Luồng xử lý

1. Đọc report manifest, mỗi `report_id` phải duy nhất.
2. Chọn patient hoặc giới hạn số report.
3. Chạy extractor theo method.
4. Lưu một dòng cho mỗi `(report_id, target)`.
5. Ghi audit gồm output extractor, confidence và quyết định adjudication.
6. Merge các shard phân tán, kiểm tra không thiếu/không trùng target.
7. Tạo QC summary: accepted, abstained, no-result và disagreement.

## Output

`outputs/silver_label/<experiment_id>/`:

- `labels.parquet`
- `audit.jsonl`
- `qc.json`
- `state/`, `result.json`, `logs/`

## Chạy

```bash
python tools/silver_labels/generate_silver_labels.py \
  --config configs/runs/00_data/silver/hybrid.yaml \
  --max-reports 100
```

## Checklist lỗi

- Không dùng dòng `abstained` hoặc `no_result` làm nhãn 0.
- Kiểm tra `report_id` duy nhất trước khi chạy.
- Provider phải có đúng `model_id` và `provider_factory`.
- Kiểm tra VRAM vì provider được load trên mỗi GPU rank.
- Diagnosis anatomy chỉ đọc accepted silver rows; silver không dùng làm evaluation label.

