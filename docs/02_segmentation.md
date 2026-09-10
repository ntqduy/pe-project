# 02. Quy trình segmentation

## Mục tiêu

Tạo pseudo-anatomy masks cho mỗi CTPA study. Đây là inference bằng public model, không phải train segmentation model.

## Luồng xử lý

1. Đọc `manifests/ctpa.csv`.
2. Resolve `image_path` tương đối với dataset root.
3. Chạy TotalSegmentator theo study, có thể shard qua nhiều GPU.
4. Chạy LungMask độc lập để cross-check phổi.
5. QC từng mask: status, volume, voxel, shape, spacing, orientation, affine.
6. Tính Dice giữa TotalSegmentator lung và LungMask.
7. Ghi state để resume, preview overlay và manifest kết quả.

## Output

`outputs/segmentation/SEG_pseudo_anatomy/` gồm:

- `masks/<study_id>/canonical/*.nii.gz`
- `manifest.parquet`: một dòng cho mỗi `(study_id, anatomy)`
- `state/<study_id>.json`, `previews/`, `result.json`, `logs/`

Các nhóm chính: lung, heart, myocardium, mediastinum, central PA, lung vessels, airways, body, RV/LV/RA/LA và các mask derived.

## Chạy

```bash
python tools/create_masks/generate_masks.py \
  --config configs/runs/00_data/segmentation.yaml \
  --max-cases 10 --gpus 0
```

Full run phải dùng `--allow-full`; muốn chạy lại sạch dùng `--overwrite`.

## Checklist lỗi

- Đường dẫn TotalSegmentator repo/weights và LungMask checkpoint phải tồn tại.
- `FAIL`, `SUSPICIOUS`, `UNAVAILABLE` không được âm thầm coi là mask hợp lệ.
- Dice thấp chỉ là cảnh báo/QC; cần quyết định rõ có loại study hay không.
- ROI và diagnosis anatomy chỉ chạy sau khi segmentation manifest đã hoàn tất.

