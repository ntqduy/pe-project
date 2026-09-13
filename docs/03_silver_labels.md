# 03. Silver labels

Stage này đọc report và tạo nhãn phụ; không dùng silver label làm ground truth đánh giá.
Bảy method dùng chung một implementation: `rule`, `falcon`, `medgemma`, `rule_falcon`,
`rule_medgemma`, `falcon_medgemma`, `rule_falcon_medgemma`.

## Cơ chế quyết định và abstention

1. Rule chỉ accept khi pattern rõ ràng; không có evidence thì chuyển stage kế tiếp hoặc
   `no_result`.
2. Falcon/MedGemma chỉ accept giá trị hợp lệ khi confidence đạt threshold cấu hình.
3. Nếu hai model được dùng để adjudicate, chúng phải trả cùng giá trị và cả hai confidence
   đạt `agreement_confidence_threshold`; disagreement hoặc confidence thấp đều `abstained`.
4. `abstained` và `failed/no_result` luôn có final value rỗng, không biến thành nhãn âm.
5. Threshold mặc định hiện là fixed pending validation calibration. Muốn báo cáo lâm sàng
   phải hiệu chỉnh trên validation và điền artifact calibration; tuyệt đối không tune trên test.

## Output chính

Ví dụ:
`E:\PE_NU\source\pe-project\outputs\silver_label\rule_falcon_medgemma\`.

```text
rule_falcon_medgemma/
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
# wrapper: một script cho mỗi cascade
MAX_REPORTS=10 bash scripts/2_silver_label/rule.sh
ALLOW_ALL=1 GPUS=0,1 bash scripts/2_silver_label/rule_falcon_medgemma.sh

# run.py trực tiếp
python run.py run data.silver.rule --max-reports 10
python run.py run data.silver.rule_falcon_medgemma --gpus 0 --allow-full
```

Bảy script trong `scripts/2_silver_label/` tương ứng bảy method: `rule.sh`, `falcon.sh`,
`medgemma.sh`, `rule_falcon.sh`, `rule_medgemma.sh`, `falcon_medgemma.sh`,
`rule_falcon_medgemma.sh`.

Sau chạy, kiểm tra tỷ lệ abstention theo target/source, lọc `abstain_reason`, đọc evidence
và so report gốc. `result.json` chỉ dùng để xem thống kê nhanh; audit ở mức case phải đọc
hai CSV chính và config snapshot.
