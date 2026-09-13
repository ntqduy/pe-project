# 12. Migration guide

| Cũ | Mới | Tương thích |
|---|---|---|
| `ROI2.nii.gz` | `ROI2_strict_heart.nii.gz` | rebuild ROI; reader dùng manifest |
| ROI `manifest.parquet` | `roi_manifest.csv` | output cũ giữ để audit |
| `logs.txt`/`logs/train.log` | `logs/run.log` | log cũ không tự xóa |
| `labels.jsonl` | `silver_labels.csv` | downstream mới đọc CSV |
| `audit.jsonl`/`qc.jsonl` | confidence CSV + QC CSV | state mới ở `.state/` |
| `all_patient` | `all_comers` | launcher còn nhận alias cũ |
| `PE_positive` | `pe_positive_only` | launcher còn nhận alias cũ |
| `image_report` | `c0` | resolver còn map alias |
| checkpoint không sidecar | `*.ckpt.metadata.json` | checkpoint cũ vẫn load nếu lineage hợp lệ |

Migration an toàn: unit test -> rebuild một patient bằng run ID mới -> so mask/QC/preview ->
chạy training smoke khi real dependency sẵn sàng -> chuyển consumer -> dùng `rg` xác nhận hết
reference -> sau đó mới archive/xóa output cũ. Không remap ROI1/3/7 như một rename.
