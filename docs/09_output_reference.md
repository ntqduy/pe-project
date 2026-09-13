# 09. Tra cứu output

Mọi path là tương đối với output root đã resolve. Ví dụ local:
`E:\PE_NU\source\pe-project\outputs\`; đường dẫn tuyệt đối thật luôn nằm trong
`resolved_config.yaml`, manifest và QC CSV.

| Stage | Dữ liệu chính | QC/log | Metadata nội bộ |
|---|---|---|---|
| segmentation | `masks/<patient>/<study>/canonical/*.nii.gz`, `manifest.parquet` | `previews/`, `logs/segmentation_qc.csv`, `logs/run.log` | `state/`, config/result |
| ROI | `rois/<patient>/<study>/<semantic>.nii.gz`, `roi_manifest.csv` | `previews/`, `logs/roi_qc.csv`, `logs/run.log` | `state/`, config/result |
| silver | `silver_labels.csv`, `silver_label_confidence.csv` | `logs/silver_label_qc.csv`, `logs/run.log` | `.state/`, config/result |
| training | `best.ckpt`, `best.ckpt.metadata.json` | `logs/history.csv`, `logs/run.log` | config, lineage, environment, metrics, result |
| evaluation | `predictions.parquet` | metrics/CI trong result | `result.json` |

- `resolved_config.yaml`: config sau merge/override/env/validation.
- `lineage.json`: provenance của best checkpoint.
- `environment.json`: runtime/package/GPU/git context.
- `metrics.json`: evaluation summary.
- `result.json`: summary gọn; không thay thế prediction/manifest.
- `history.csv`: epoch, loss, primary metric, LR, time, VRAM, task losses.
- `predictions.parquet`: patient/study, ground truth và probability dùng để audit/bootstrap.

Layout silver cũ `labels.jsonl`, `audit.jsonl`, `qc.jsonl` không còn được sinh. Không xóa
output lịch sử trước khi xác nhận không consumer nào còn tham chiếu.
