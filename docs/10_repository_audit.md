# 10. Repository audit

Audit được thực hiện trước khi sửa code; bảng này ghi hiện trạng tìm thấy và quyết định.

| Thành phần | Hiện trạng trong code | Vấn đề | Thay đổi đã thực hiện |
|---|---|---|---|
| ROI mapping | ROI1 heart+mediastinum; ROI2 strict heart; ROI3 dilated central PA; ROI4 PA tree; ROI5 lung; ROI6 lung-vessels; ROI7 heart+hilar; ROI8 control ROI2/4/6 | tên file chỉ có mã; yêu cầu ví dụ ROI1 full CT không khớp code | giữ mapping thật, thêm semantic registry/file/manifest; không remap ROI |
| ROI8 | chọn các voxel hợp lệ gần random anchor | giữ volume nhưng không giữ hình; chưa loại full forbidden union | rigid translation, exact voxel/physical volume, forbidden union, seed, explicit failure |
| Preview | 1-2 slice và metadata hạn chế | khó review theo patient/anatomy | 1-3 slice deterministic, CTPA W/L, colored overlay và metadata |
| Logging/QC | nhiều tên log/status khác nhau | khó tổng hợp và thiếu traceback nhất quán | `logs/run.log`, QC CSV schema chung, canonical statuses |
| Silver | `labels/audit/qc.jsonl`; MedGemma accept non-null | phân tán; confidence gate chưa đủ | hai CSV chính, QC log, threshold cho từng model/agreement, abstention |
| PEFT | DAPT/alignment full; probe frozen; diagnosis/prognosis LoRA thật | config chưa có interface `finetuning.strategy`; partial chưa có | map full/linear_probe/lora; reject partial; log trainable parameters |
| Checkpoint | atomic state+lineage, có module transfer | load report bị bỏ, shape mismatch khó đọc, thiếu sidecar | structured load report và metadata SHA-256 sidecar; thêm diagnosis init |
| Diagnosis | native labels là PE positive/acute/subsegmental | thiếu mode contract rõ trong matrix | single/multitask schema validator, exact labels, launcher overrides đồng bộ |
| Prognosis | hai cohort legacy, bảy endpoints đã có | tên protocol và inclusion/exclusion chưa rõ | canonical aliases và cohort membership audit |
| RSPECT/Turkey | RSPECT transfer có config; Turkey external theo proposal chưa có | thiếu Turkey segmentation/ROI/test-only; external threshold từng lấy sai cohort | thêm Turkey contracts; khóa threshold từ INSPECT validation; giữ blocked |
| Configs | nhiều component YAML cực ngắn | khó đọc và lặp | gom thành 8 catalog; run files vẫn là entrypoint đọc được |

## Kết luận audit quan trọng

- File `configs/clinical/ehr_prohibited_features.json` chỉ chứa policy nhỏ, được đọc gián
  tiếp bởi EHR preprocessing nhưng tổ chức quá sâu. Policy đã được gom vào config clinical
  thực sự dùng; file cũ xóa sau khi `rg` xác nhận không reference.
- ROI3/ROI7 vẫn có consumer counterfactual, nên không xóa. ROI1 không phải full CT.
- LoRA diagnosis/prognosis là implementation thật: inject module, freeze base, optimizer chỉ
  nhận trainable parameter. `partial` chưa có.
- Prognosis endpoint thật không gồm ICU/mechanical ventilation/hemodynamic deterioration;
  không thêm endpoint không có ground truth.
- RSPECT transfer/Turkey external chưa thể chạy vì thiếu normalized data và real
  weights; không được báo là đã thành công.
