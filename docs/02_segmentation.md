# 02A. Segmentation pseudo-anatomy

Stage này chỉ tạo mask giải phẫu bằng TotalSegmentator và QC chéo phổi bằng LungMask; nó
không train model segmentation. INSPECT dùng `data.segmentation`; RSPECT dùng
`data.segmentation.rspect` sau khi external manifest đã được chuẩn hóa.

## Cách chạy

```bash
# wrapper
python run.py run data.segmentation --max-cases 1 --gpus 0
python run.py run data.segmentation --allow-full --gpus 0,1
python run.py run data.segmentation --allow-full

# run.py trực tiếp
python run.py preflight data.segmentation
python run.py run data.segmentation --gpus 0 --max-cases 1
python run.py run data.segmentation --gpus 0,1 --allow-full
```

Chỉ chạy TotalSegmentator, không chạy LungMask:

```bash
LUNGMASK=0 PROFILE=smoke_30 GPUS=0 bash scripts/run_segmentation.sh
python run.py run data.segmentation --gpus 0 --allow-full --set segmentation.lungmask.enabled=false
```

Mọi mask dùng downstream vẫn đến từ TotalSegmentator nên giống hệt. Chỉ mất `lung_lungmask`
(dòng `UNAVAILABLE` trong `qc_summary.csv`, không tính là study lỗi) và bước QC Dice phổi chéo
model, nên mask phổi của TotalSegmentator không còn bị hạ xuống SUSPICIOUS/FAIL khi hai model
lệch nhau. Chọn trước khi bắt đầu một run: chạy tiếp run đó với thiết lập kia bị từ chối vì
resolved config không còn khớp.

Hai study độc lập có thể chạy song song. Mỗi GPU nhận một worker; ROI có thể chạy song
song bằng CPU sau khi segmentation của các study tương ứng đã hoàn tất và manifest đã được
ghi.

## Output

Ví dụ tuyệt đối với run INSPECT:
`E:\PE_NU\source\pe-project\outputs\segmentation\SEG_pseudo_anatomy\`.
Nếu `PE_CLOUD_ROOT` trỏ nơi khác, xem đường dẫn thật trong `resolved_config.yaml`.

```text
SEG_pseudo_anatomy/
  masks/<patient_id>/<study_id>/<anatomy>.nii.gz   # 26 mask/study, một tầng thư mục
  previews/<patient_id>_<study_id>.png             # ảnh tổng hợp của 20 study đầu manifest để kiểm tra bằng mắt
  qc_summary.csv                                   # 1 dòng/(study, anatomy), cột phẳng
  manifest.parquet                                 # chỉ mục mask cho ROI builder
  logs/run.log
  state/<patient_id>/<study_id>.json
  resolved_config.yaml
  result.json
```

- `masks/<patient>/<study>/*.nii.gz`: 26 mask nhị phân dùng downstream (lung, lung_left,
  lung_right, lung_upper_lobe_left, lung_lower_lobe_left, lung_upper_lobe_right,
  lung_middle_lobe_right, lung_lower_lobe_right, heart, strict_heart, myocardium, mediastinum,
  central_pa, lung_arteries, lung_veins, lung_vessels, airways, pa_tree, hilar_vessels, body,
  body_wall, rv, lv, ra, la, lung_lungmask). Năm thuỳ phổi lấy từ cùng lần chạy `total` dựng
  nên `lung` (không tốn thêm inference); `lung_left`/`lung_right` là hợp các thuỳ mỗi bên. Giữ
  lại vì impression khu trú PE theo bên (~88% báo cáo PE dương tính) và thuỳ (~62%); lingula,
  phân thuỳ và cấp động mạch (segmental/subsegmental) không có mask. Đây là dữ
  liệu chính. Output thô của từng task TotalSegmentator (`total`, `trunk_cavities`,
  `heartchambers_highres`, `lung_vessels`, `body`) chỉ là file trung gian: được ghi vào thư mục
  tạm local (`segmentation.scratch_dir`, mặc định `/tmp`) và bị xoá sau khi dựng xong mask, nên
  không còn thư mục `tasks/` hay `canonical/`. Run cũ còn `tasks/` + `canonical/` là do bị ngắt
  giữa chừng; lần chạy lại study đó sẽ tự dọn.
- `previews/<patient>_<study>.png`: một contact sheet cho mỗi study trong 20 study đầu của
  manifest (`segmentation.preview_cases: 20`, luôn là cùng nhóm kể cả khi chạy tiếp run bị ngắt;
  `null` = mọi study). Vẽ ảnh chiếm ~2/10 phút mỗi study trên smoke_30, nên full cohort chỉ vẽ
  một mẫu. Mỗi panel là lát axial
  có diện tích mask lớn nhất (phía trước ở trên, bên trái bệnh nhân ở bên phải ảnh, z đếm từ
  đầu dưới), overlay cam, tiêu đề `anatomy | z | mL | status`. Mask rỗng hiện chữ đỏ `EMPTY`,
  mask không tạo được hiện `UNAVAILABLE` + lý do, mask trùng hệt mask khác ghi `= <anatomy>`.
  Panel cuối là lát coronal tổng hợp mọi mask với tỉ lệ đúng theo spacing. Ảnh được ghi ngay
  sau khi TotalSegmentator xong (tiêu đề ghi "LungMask pending") và ghi lại sau LungMask, nên
  run bị ngắt vẫn có ảnh để kiểm tra.
- `qc_summary.csv`: file mở đầu tiên khi kiểm tra mask. Cột: `patient_id, study_id, anatomy,
  status, voxel_count, volume_ml, n_components, z_first, z_last, duplicate_of,
  cross_model_dice, reason, preview_png, mask_path`. `voxel_count=0` nghĩa là mask rỗng;
  `duplicate_of` khác rỗng nghĩa là hai mask giống hệt nhau từng voxel (ví dụ `pa_tree` =
  `lung_arteries` khi mask lung_arteries của TotalSegmentator đã chứa central PA).
- `manifest.parquet`: một dòng cho `(patient_id, study_id, anatomy)`, gồm đường dẫn CT và
  mask, backend/source, status, voxel/volume, geometry, hash mask và QC chéo nếu có.
- `logs/run.log`: bản ghi terminal có UTC timestamp, lệnh, config, thời gian từng task
  TotalSegmentator và từng mask dẫn xuất, lỗi.
- `state/`: cache resume nội bộ; không dùng làm kết quả khoa học.
- `result.json`: tổng số requested/processed/failed/partially_failed,
  `partial_failure_study_ids`, `empty_masks`, `duplicate_masks`, backend, compute và
  reproducibility. Nếu mask chính có `FAIL` hoặc `UNAVAILABLE`, status là `completed_with_failures` và
  lệnh trả mã lỗi; lần chạy tiếp theo sẽ thử lại study có `UNAVAILABLE`.
  ROI chỉ nhận status `completed`.

`body_wall` và `hilar_vessels` được tính bằng distance transform chính xác theo từng slab
trục z (`source/imaging/morphology.py`). Bản cũ dùng `binary_erosion`/`binary_dilation` với
khối cầu 15/12 mm; scipy cấp bảng offset ~14.7 GB cho CTPA 512×512 nên bị OOM trên VM 16 GB và
study không bao giờ tới bước preview/QC.

TotalSegmentator chạy với `totalseg_resample_threads=1`, `totalseg_saving_threads=1`
(mặc định của nó là 6 tiến trình lưu, mỗi tiến trình vài GB → bị OOM-kill trên VM 16 GB và
nnU-Net treo mãi). Mỗi task có `task_timeout_sec` (mặc định 3600 s): quá hạn thì task và các
worker bị kill, mask tương ứng thành `UNAVAILABLE` thay vì treo cả run.

Status QC trong manifest là `PASS`, `SUSPICIOUS`, `FAIL`, `UNAVAILABLE`.

## Kiểm tra nhanh

1. Mở `qc_summary.csv`, lọc `status != PASS`, `voxel_count = 0` và `duplicate_of` khác rỗng.
2. Mở `previews/<patient>_<study>.png` của vài study: mọi panel phải có overlay đúng cơ quan.
3. Không coi anatomy `UNAVAILABLE` là mask rỗng hợp lệ.
4. Chỉ truyền mask có geometry khớp CT sang ROI.
5. Run cũ theo layout `canonical/`: chạy lại với `OVERWRITE=1` (config đã đổi nên resume sẽ bị
   chặn bởi kiểm tra config hash).
