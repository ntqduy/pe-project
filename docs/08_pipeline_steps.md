# 08. Thứ tự pipeline và chạy song song

```text
dataset build
  -> segmentation ---------> ROI
  -> silver labels
  -> DAPT -> alignment/C0 -> diagnosis -> prognosis
  -> RSPECT supervised transfer (optional)
  -> Turkey normalization -> segmentation/ROI -> primary test-only evaluation
```

Segmentation các study chạy song song theo GPU. ROI các study hoàn tất chạy song song theo
CPU. Silver report sharding cũng độc lập. Các nhánh này có thể chạy đồng thời miễn là input
của từng case đã tồn tại và không ghi cùng run directory.

Trước stage model:

```bash
python run.py plan <experiment>
python run.py preflight <experiment>
```

Trước full run, dùng `--max-cases 1` hoặc profile `smoke_30`, kiểm tra PNG/QC/log rồi mới
`--allow-full`. Không sửa status thủ công để vượt `failed/UNAVAILABLE`.

## RSPECT và Turkey

Tên chuẩn trong repo là `RSPECT`; nguồn release được mô tả là RSNA-STR Pulmonary Embolism
Detection. Theo email, RSPECT là nguồn public cho pretraining/supervised transfer. Chuỗi này:

1. giải nén và tạo `/mnt/RSPECT_dataset/manifests/rsna_diagnosis.csv` với patient-level
   train/validation/test;
2. chạy `data.segmentation.rspect`;
3. chạy `data.roi.rspect` nếu model cần anatomy;
4. train/freeze encoder transfer theo config `repr.rspect.*` nếu dùng nhánh này.

Turkey mới là external test chính của proposal. Sau khi nhận dữ liệu, normalize thành
`/mnt/Turkey_dataset/manifests/turkey_diagnosis.csv`, chạy `data.segmentation.turkey`,
`data.roi.turkey`, rồi evaluate `diag.external.turkey_test`.

Config external evaluation không thể chạy bằng `train_task.py`. Threshold chỉ được chọn trên
INSPECT validation, khóa trong source `result.json`, rồi áp dụng một lần lên Turkey/RSPECT
test. Hiện normalized data chưa có nên registry đánh dấu các stage này `blocked`.

## Blocker còn lại

- backbone factory/output adapter và real checkpoint cần được điền sau khi inspect weight;
- RSPECT và Turkey normalized manifests chưa sẵn sàng;
- EHR columns và PESI clinical approval còn thiếu cho một số prognosis arms;
- contour cần expert-reviewed embolus masks.

