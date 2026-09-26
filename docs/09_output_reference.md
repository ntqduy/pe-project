# 09. Tra cứu output

Mọi path là tương đối với output root đã resolve. Ví dụ local:
`E:\PE_NU\source\pe-project\outputs\`; đường dẫn tuyệt đối thật luôn nằm trong
`resolved_config.yaml`, manifest và QC CSV.

| Stage | Dữ liệu chính | QC/log | Metadata nội bộ |
|---|---|---|---|
| segmentation | `masks/<patient>/<study>/<anatomy>.nii.gz`, `manifest.parquet` | `previews/<patient>_<study>.png`, `qc_summary.csv`, `logs/run.log` | `state/`, config/result |
| ROI | `rois/<patient>/<study>/<semantic>.nii.gz`, `roi_manifest.csv` | `previews/`, `logs/roi_qc.csv`, `logs/run.log` | `state/`, config/result |
| silver | `silver_labels.csv`, `silver_label_confidence.csv` | `logs/silver_label_qc.csv`, `logs/run.log` | `.state/`, config/result |
| train + evaluate (diagnosis/prognosis) | `epoch_<N>/result.csv`, `epoch_<N>/checkpoint/{best,last}.ckpt` | `epoch_<N>/logs.txt`, `epoch_<N>/training_curves.png`, `epoch_<N>/preview/*.{html,png}`, `epoch_<N>/predictions.csv` | `epoch_<N>/resolved_config.yaml`, `epoch_<N>/result.json` |
| zero-shot baseline | `result.csv` (cùng cột với train/evaluate) | `logs.txt`, `predictions.csv`, `preview/*.{html,png}` | `resolved_config.yaml`, `result.json` |

- `epoch_<N>/` (train + evaluate): `N` là ngân sách `training.epochs` (`EPOCHS=` của script),
  không phải số epoch thực chạy. Mỗi ngân sách là một run độc lập, có đủ config/result riêng,
  nên `EPOCHS=1` và `EPOCHS=30` của cùng run ID nằm cạnh nhau (`<RUN_ID>/epoch_1/`,
  `<RUN_ID>/epoch_30/`). Chỉ chạy lại đúng ngân sách đã có mới bị preflight chặn
  (`OUTPUT/collision`); `--overwrite`/`OVERWRITE=1` chỉ xóa đúng `epoch_<N>/` đó. Dừng sớm
  vẫn ghi vào `epoch_<N>/`; số epoch thực chạy nằm trong `result.json` (`epochs_run`,
  `stopped_early`) và tiêu đề `training_curves.png`.
- `resolved_config.yaml`: config sau merge/override/env/validation.
- `result.json`: payload đầy đủ cho máy đọc (metric + CI của mọi metric, lineage, compute);
  `tools/build_summary.py` đọc file này.
- `epoch_<N>/result.csv`: bảng metric cho người đọc, 1 dòng/(target, split), cùng cột cho mọi
  run nên có thể ghép nhiều run để so sánh:
  `experiment, target, split, n_patients, n_studies, n_pos, n_neg, auroc, auroc_ci_low,
  auroc_ci_high, auprc, auprc_ci_low, auprc_ci_high, threshold, threshold_rule, sensitivity,
  specificity, ppv, npv, f1, balanced_accuracy, accuracy, brier, [calibration_intercept,
  calibration_slope — chỉ prognosis], note`.
  - `threshold`: ngưỡng xác suất, `p ≥ threshold` → dự đoán dương. Chỉ ảnh hưởng sensitivity,
    specificity, PPV, NPV, F1, accuracy, balanced accuracy; AUROC/AUPRC/Brier không phụ thuộc.
  - `threshold_rule`: `youden_on_validation` = chọn trên validation bằng Youden J (max
    sensitivity + specificity − 1) rồi áp nguyên cho train/validation/test;
    `default_0.5_validation_one_class` = validation của target đó chỉ có 1 lớp nên không chọn
    được, dùng 0.5; `locked_internal_validation` = external test dùng ngưỡng đã khoá từ
    validation nội bộ. Trong `result.json` field cũ `threshold_source` vẫn giữ giá trị
    `validation` / `fallback_0.5_validation_single_class` / `internal_validation_artifact`.
  - Ô trống = không tính được; cột `note` ghi lý do: split chỉ có 1 lớp → AUROC/AUPRC không xác
    định; không có ca nào bị đoán dương/âm → PPV/NPV không xác định; threshold nằm ngoài khoảng
    `y_prob` → mọi ca bị đoán cùng 1 lớp; `n_pos`/`n_neg` < 5 → ước lượng không ổn định;
    calibration cần ≥ `evaluation.calibration_min_events` (mặc định 10) ca mỗi lớp.
  - CI 95% là patient bootstrap, chỉ có ở dòng `test` (theo protocol). CI của mọi metric nằm
    trong `result.json`.
- `epoch_<N>/predictions.csv`: `split, target, patient_id, study_id, y_true, y_prob, y_pred`
  cho train/validation/test; dùng để audit metric và làm `--reference-predictions`.
- `epoch_<N>/logs.txt`: log terminal của train rồi evaluate (evaluate ghi nối tiếp): header,
  từng epoch, threshold từng target, số ca dương/âm từng split, cảnh báo, block FINAL TEST.
- `epoch_<N>/training_curves.png`: loss và AUROC train/validation theo epoch; đường chấm dọc
  là epoch được lưu `best.ckpt`. Ở cả hai ô, train = nét đứt + chấm rỗng, validation = nét liền +
  chấm đặc. Một target (diagnosis, hoặc prognosis có `TARGET`): train xanh dương, validation cam
  như ô loss. Nhiều target (prognosis 7 endpoint): màu là target, kiểu nét là split.
- `epoch_<N>/preview/NN_<patient>_<study>_<TP|TN|FP|FN>.{html,png}`: Grad-CAM của logit target
  chính cho tối đa 5 study đầu trong validation (không chọn theo đúng/sai). `.html` là viewer
  offline: thanh trượt qua mọi lát input model nhận (CT-FM cached: 10 lát feature, CT chỉ là
  ảnh tham chiếu ở tâm ô feature), mỗi lát CT | CT + Grad-CAM, opacity, montage
  8 slice điểm CAM cao nhất, slice gốc NIfTI, hướng A/P/R/L và mục kỹ thuật. `.png` là bản tĩnh
  gồm header và montage. Header ghi ground truth, predicted, xác suất, threshold của evaluate
  và `p − threshold`. CAM được chuẩn hóa một lần cho cả volume; không phải xác suất từng
  pixel/slice hay mask huyết khối. Chi tiết và giới hạn: `docs/05_diagnosis_training.md`, mục
  "Preview Grad-CAM". Tạo lại mà không evaluate lại: `tools/tasks/gradcam_preview.py`.
- `RUN/preview/NN_<patient>_<study>_<TP|TN|FP|FN>.{html,png}` (thư mục run PENet zero-shot):
  cùng viewer như trên cho 5 study đầu của validation. Tool chạy lại PENet và lấy Grad-CAM của
  logit xác suất series tại `encoders[-1]`, gradient đi qua bước gộp cửa sổ (max: chỉ cửa sổ quyết
  định có CAM). `result.json` có khóa `preview` ghi trạng thái CAM và độ lệch xác suất. Tạo lại
  cho run có sẵn: `python -m source.imaging.penet_preview --run-dir <run> --raw-root <CT/full/CTPA>`.

Layout silver cũ `labels.jsonl`, `audit.jsonl`, `qc.jsonl` không còn được sinh. Không xóa
output lịch sử trước khi xác nhận không consumer nào còn tham chiếu.
