# 02A. Segmentation pseudo-anatomy

Stage này chỉ tạo mask giải phẫu bằng TotalSegmentator và QC chéo phổi bằng LungMask; nó
không train model segmentation. INSPECT dùng `data.segmentation`; RSPECT dùng
`data.segmentation.rspect` sau khi external manifest đã được chuẩn hóa.

## Cách chạy

```bash
# wrapper
MAX_CASES=1 GPUS=0 bash scripts/1_segmentation/totalsegmentator.sh
ALLOW_ALL=1 GPUS=0,1 bash scripts/1_segmentation/totalsegmentator.sh
ALLOW_ALL=1 SET="segmentation.workers=8" bash scripts/1_segmentation/totalsegmentator.sh

# run.py trực tiếp
python run.py preflight data.segmentation
python run.py run data.segmentation --gpus 0 --max-cases 1
python run.py run data.segmentation --gpus 0,1 --allow-full
```

Hai study độc lập có thể chạy song song. Mỗi GPU nhận một worker; ROI có thể chạy song
song bằng CPU sau khi segmentation của các study tương ứng đã hoàn tất và manifest đã được
ghi.

## Output

Ví dụ tuyệt đối với run INSPECT:
`E:\PE_NU\source\pe-project\outputs\segmentation\SEG_pseudo_anatomy\`.
Nếu `PE_CLOUD_ROOT` trỏ nơi khác, xem đường dẫn thật trong `resolved_config.yaml`.

```text
SEG_pseudo_anatomy/
  masks/<patient_id>/<study_id>/canonical/<anatomy>.nii.gz
  previews/<patient_id>/<study_id>/segmentation/<anatomy>/
    <anatomy>_axial_slice_001_z####.png
    <anatomy>_axial_slice_002_z####.png
    <anatomy>_axial_slice_003_z####.png
  manifest.parquet
  logs/run.log
  logs/segmentation_qc.csv
  state/<patient_id>/<study_id>.json
  resolved_config.yaml
  result.json
```

- `masks/.../canonical/*.nii.gz`: mask nhị phân dùng downstream. Đây là dữ liệu chính.
- `manifest.parquet`: một dòng cho `(patient_id, study_id, anatomy)`, gồm đường dẫn CT và
  mask, backend/source, status, voxel/volume, geometry và QC chéo nếu có.
- `previews/`: tối đa ba axial slice cho anatomy được cấu hình: slice lớn nhất và hai slice
  có ngữ cảnh trước/sau. PNG ghi patient, study, anatomy, z-index, source, QC, voxel,
  thể tích và cửa sổ CTPA `W/L=700/100`.
- `logs/run.log`: bản ghi terminal có UTC timestamp, lệnh, config, tiến độ, lỗi và thời gian.
- `logs/segmentation_qc.csv`: schema QC chung: `run_id, patient_id, study_id, stage,
  item_name, status, input_path, output_path, qc_checks, failure_reason, timestamp`.
- `state/`: cache resume nội bộ; không dùng làm kết quả khoa học.
- `result.json`: tổng số requested/processed/failed, backend, compute và reproducibility.

Status QC chuẩn là `pass`, `failed`, `skipped`, `abstained`. Manifest cũ có
`PASS/SUSPICIOUS/FAIL/UNAVAILABLE` vẫn đọc được trong giai đoạn migration.

## Kiểm tra nhanh

1. Mở PNG của `lung`, `strict_heart`, `pa_tree` cho vài patient.
2. Lọc `logs/segmentation_qc.csv` theo `status=failed`.
3. Không coi anatomy `UNAVAILABLE` là mask rỗng hợp lệ.
4. Chỉ truyền mask có geometry khớp CT sang ROI.
