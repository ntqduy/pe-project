# 02B. ROI và control ROI

ROI được dựng từ mask segmentation đã lưu; stage này không chạy lại TotalSegmentator.
Tên file luôn chứa mã và ý nghĩa để người đọc không phải nhớ ROI số mấy là gì.

## ROI contract đã audit

| Mã | Tên semantic | Công thức hiện có | Vai trò |
|---|---|---|---|
| ROI1 | `heart_mediastinum` | heart union mediastinum | giữ vùng |
| ROI2 | `strict_heart` | strict heart | giữ vùng |
| ROI3 | `dilated_central_pulmonary_artery` | central PA dilation | loại vùng |
| ROI4 | `pulmonary_artery_tree` | PA tree | giữ vùng |
| ROI5 | `whole_lung_with_vessels` | toàn phổi | giữ vùng |
| ROI6 | `lung_parenchyma_without_large_vessels` | lung trừ vessel lớn | giữ vùng |
| ROI7 | `heart_hilar_vessels_exclusion` | heart union hilar vessels | loại vùng |
| ROI8 | `control_for_<source>` | rigid translation của ROI2/4/6 | control |

ROI3 và ROI7 được giữ vì counterfactual removal đang dùng chúng. Không đổi ROI1 thành
“full CT”: full CTPA đã là input baseline, và đổi nghĩa ROI1 sẽ làm sai lịch sử thí nghiệm.

## ROI8

ROI8 dùng seed ổn định từ `(seed, patient_id, study_id, source_roi)`. Thuật toán chỉ dịch
chuyển cứng mask nguồn, không co, crop, resample, fill voxel gần nhất hoặc đổi hình. Control:

- nằm trong body/valid CT;
- không giao source ROI;
- không giao union anatomy cấm (`heart`, `mediastinum`, `central_pa`, pulmonary vessels,
  hilar vessels) sau margin vật lý;
- giữ đúng voxel count và thể tích vật lý;
- mặc định giữ nguyên z-range để giảm bias theo trục đầu-chân.

Nếu không có vị trí thỏa điều kiện trong `max_attempts`, ROI8 được ghi `failed` với
`failure_reason=no_valid_control_location`; tuyệt đối không ghi mask rỗng như một thành công.

## Output

```text
E:\PE_NU\source\pe-project\outputs\roi\ROI_anatomy_and_controls\
  rois/<patient_id>/<study_id>/ROI1_heart_mediastinum.nii.gz
  rois/<patient_id>/<study_id>/ROI2_strict_heart.nii.gz
  ...
  rois/<patient_id>/<study_id>/ROI8_control_for_ROI4_pulmonary_artery_tree.nii.gz
  previews/<patient_id>/<study_id>/rois/<semantic_name>/*.png
  roi_manifest.csv
  logs/run.log
  logs/roi_qc.csv
  state/<patient_id>/<study_id>.json
  resolved_config.yaml
  result.json
```

`roi_manifest.csv` là index chính, một dòng cho mỗi ROI hoặc mỗi `(ROI8, source ROI)`. Các
cột quan trọng: `patient_id, study_id, roi_code, roi_name, source_roi, operation, status,
qc_severity, mask_path, image_path, generation_seed, voxel_count, physical_volume_mm3,
failure_reason`. Với ROI8 còn có translation, overlap/intersection, Dice với source, exact
voxel/physical-volume match và số attempt. Đường dẫn output được ghi tuyệt đối.

`logs/roi_qc.csv` dùng schema QC chung và chứa chi tiết geometry/body containment/overlap.
Preview dùng cùng quy tắc ba slice và W/L như segmentation.

## Chạy

```bash
python run.py run data.roi --max-cases 1
python run.py run data.roi --allow-full
```

Tăng `roi.workers` để xử lý nhiều study CPU song song. Có thể chạy segmentation và ROI
song song theo batch/study, nhưng ROI của một study chỉ được bắt đầu sau khi mask canonical
của study đó đã hoàn tất.
