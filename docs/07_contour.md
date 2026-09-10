# 07. Quy trình contour

## Trạng thái

Contour là stage deferred, chưa thuộc active pipeline.

## Mục tiêu

Localize/refine embolus mask, khác với diagnosis là chỉ dự đoán PE presence.

## Luồng xử lý dự kiến

1. Có manifest `manifests/contour.csv` với `embolus_mask_path`.
2. Load CT và reviewed embolus mask.
3. Load image encoder từ checkpoint diagnosis đã được chỉ định.
4. Train decoder contour.
5. Đánh giá Dice/IoU và lưu prediction, checkpoint, QC.

## Điểm đang chặn

- Manifest contour và embolus masks chưa được cung cấp.
- Config hiện trỏ `lineage.source_experiment: DX01`, là ID cũ không còn active.
- Cần quyết định checkpoint diagnosis active nào sẽ làm initialization trước khi chạy.

Không tự sửa checkpoint source chỉ để làm preflight xanh; đây là quyết định về thiết kế thí nghiệm.

