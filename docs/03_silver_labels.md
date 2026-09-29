# 03. Silver labels

Stage này đọc report và tạo nhãn phụ; không dùng silver label làm ground truth đánh giá.
Chỉ còn một method sinh nhãn: `medgemma`. Cascade machinery trong
`source/silver/generator.py` giữ nguyên nên thêm lại một stage chỉ là sửa một dòng.

## Cơ chế quyết định và abstention

Mỗi report được hỏi 19 target, mỗi target một lần gọi MedGemma (prompt `v2`).

1. **MedGemma trả lời trước.** Prompt v2 quy định rõ: `true` khi report ghi có (mọi mức
   độ/bên đều tính, vd. "small bilateral pleural effusions"); `false` khi report ghi không có
   hoặc một câu rõ ràng loại trừ (vd. "No pulmonary embolism" → mọi vị trí PE đều false);
   `null` khi không nhắc tới hoặc chỉ nói nước đôi. Có ví dụ JSON; `max_new_tokens=384`.
2. **Parse mềm, không làm mất nhãn vì lỗi format:** `"yes"/"present"/"small"` → `true`,
   `"no"/"absent"` → `false`, confidence `"95%"` → 0.95, field thừa bị bỏ, offset evidence
   được tính lại từ câu trích (không tin offset model). Mọi chuyển đổi ghi ở `normalization`
   trong audit. Câu trích không có trong report → không accept (`medgemma_evidence_not_in_report`).
3. **Retry:** câu trả lời không dùng được (không phải JSON, value sai kiểu) được hỏi lại 1 lần
   kèm thông báo lỗi (`silver.medgemma.retries`). Raw response luôn lưu trong audit `.state/`.
4. **Chỉ accept khi confidence ≥ `medgemma_confidence_threshold` (0.8).**
5. **`rule_rescue: true` — regex (`source/silver/rules.py`) làm backup, không thay MedGemma:**
   - MedGemma chắc chắn nhưng regex thấy câu ghi rõ điều ngược lại → abstain
     (`medgemma_rule_conflict`);
   - MedGemma abstain/lỗi nhưng report ghi rõ ràng → lấy giá trị regex, `source=rule`
     (`rule_..._after_medgemma_abstained` / `_after_medgemma_failure`).
6. **`pe_consistency: true` — nhất quán trong một report:** `pe_present=false` → mọi vị trí PE
   (central/lobar/segmental/subsegmental/saddle) false, acuity bị rút; chưa quyết được
   `pe_present` thì "vị trí = false" bị abstain vì đó là đoán.
7. `abstained` và `failed/no_result` luôn có final value rỗng, không biến thành nhãn âm.
8. Threshold mặc định là fixed pending validation calibration; không tune trên test.

Giới hạn dữ liệu: INSPECT chỉ có phần **IMPRESSION** của report, không có FINDINGS. Những gì
impression không nhắc (RV, septal bowing, emphysema, ...) đúng là không trích được và sẽ
`abstained` — đó không phải lỗi parse. Đổi `prompt_version`, `rule_rescue`, `pe_consistency`
hay threshold, model checkpoint hoặc nội dung report thì cache `.state/` tự bị bỏ và report
được gán nhãn lại. Report có lỗi kỹ thuật `no_result` cũng được hỏi lại khi resume.

## Output chính

Ví dụ:
`E:\PE_NU\source\pe-project\outputs\silver_label\medgemma\`.

```text
medgemma/
  silver_labels.csv
  silver_label_confidence.csv
  logs/run.log
  logs/silver_label_qc.csv
  .state/<report_hash>.json
  resolved_config.yaml
  result.json
```

Chỉ hai CSV đầu là bảng khoa học chính. `.state` là resume cache; `resolved_config.yaml`
và `result.json` là metadata/tổng hợp, không phải bản sao bảng nhãn.

### `silver_labels.csv`

Một dòng cho `(patient_id, study_id, report_id, target)`. Các cột:

| Cột | Nội dung |
|---|---|
| `patient_id, study_id, report_id, report_hash, split` | định danh và split gốc |
| `target` | một trong 19 target canonical |
| `value` | JSON scalar: boolean, category, số hoặc `null` |
| `status` | backward-compatible: `accepted`, `abstained`, `no_result` |
| `label_status` | chuẩn mới: `accepted`, `abstained`, `failed` |
| `source`, `label_source` | rule/model/cascade đã quyết định |
| `confidence` | confidence của quyết định; không mặc định là xác suất PE dương |
| `reason` | lý do accept/abstain/fail |
| `checkpoint_id`, `run_id`, `schema_version` | lineage của extractor |

Training chỉ đọc dòng `accepted`; mọi dòng khác được mask khỏi loss.

### `silver_label_confidence.csv`

Một dòng tương ứng mỗi quyết định, gồm `task_name, predicted_label, probability,
confidence_score, positive_threshold, negative_threshold, minimum_confidence_threshold,
decision, abstain_reason, failure_reason, label_source, checkpoint_id, run_id, evidence_text`.
`probability` và positive/negative threshold để trống cho provider chỉ trả confidence của
class được chọn; code không giả confidence thành calibrated probability.

### `logs/run.log`

Ghi UTC timestamp, command, config, method, model/checkpoint đã dùng, số report,
accepted/abstained/failed, traceback khi lỗi và elapsed time.

`logs/silver_label_qc.csv` dùng schema QC chung cho từng report-target. Đây là operational
QC index; nội dung khoa học chi tiết vẫn nằm ở hai CSV chính.

## Target schema

Các target gồm PE presence/acuity/location; RV enlargement, RV/LV mention/value/abnormal,
septal bowing, reflux; pleural/pericardial effusion; malignancy-related finding; chronic
lung disease, fibrosis và emphysema. Chi tiết kiểu/range nằm ở
`source/silver/schema.py`; unknown target hoặc value sai kiểu bị từ chối.

## Chạy và QC

```bash
PROFILE=smoke_30 MAX_REPORTS=30 GPUS=0 bash scripts/tool/run_silver_labels.sh      # thử 30 report
OVERWRITE=1 PROFILE=smoke_30 MAX_REPORTS=30 GPUS=0 bash scripts/tool/run_silver_labels.sh
PROFILE=full_inspect GPUS=0,1 bash scripts/tool/run_silver_labels.sh              # toàn bộ
```

Sau chạy, kiểm tra tỷ lệ abstention theo target/source, lọc `abstain_reason`, đọc evidence
và so report gốc. `result.json` chỉ dùng để xem thống kê nhanh; audit ở mức case phải đọc
hai CSV chính và config snapshot.
