# `analysis/` — EDA cho dataset profile

Mô tả dữ liệu **đã build**, để hiểu cohort trước khi train. Chỉ đọc artifact dẫn xuất, không
sửa manifest, không tạo ra thứ gì training tiêu thụ — chạy lại lúc nào cũng được.

```bash
bash scripts/ana.sh                                  # PROFILE=test_500_sample
PROFILE=full_inspect bash scripts/ana.sh
PROFILE=smoke_30 RULE_SAMPLE=0 bash scripts/ana.sh   # 0 = quét mọi report
NO_FIGURES=1 bash scripts/ana.sh                     # chỉ số, không vẽ PNG
```

Hoặc gọi thẳng:

```bash
python analysis/run_eda.py --profile full_inspect --rule-sample 5000
```

## Output

```text
<output_root>/EDA/<profile>/
├── EDA_REPORT.md        đọc cái này trước
├── summary.json         toàn bộ số, machine-readable
├── inventory.csv        bảng nào có / thiếu / không đọc được
├── cohort.csv           patient & study theo manifest và theo split
├── labels.csv           prevalence + missingness, overall và theo split
├── outcomes.csv         event count theo endpoint + protocol gợi ý
├── rule_coverage.csv    regex giải quyết được bao nhiêu / 19 target
├── figures/*.png
└── logs/run.log
```

## Sáu phần của báo cáo

| Phần | Trả lời | Module |
|---|---|---|
| Inventory | Stage 0 đã sinh ra những bảng nào, thiếu gì | `loaders.py` |
| Cohort | Bao nhiêu patient/study, phân bố split, **có leakage không** | `cohort.py` |
| Labels | Prevalence, missingness, mất cân bằng lớp | `labels.py` |
| Outcomes | Event count mỗi endpoint, có đủ để hold-out không | `outcomes.py` |
| Geometry | Shape/spacing thật, kích thước volume cache | `geometry.py` |
| Reports | Độ dài report, regex phủ được bao nhiêu target | `reports.py` |

## Ba điều báo cáo này nói mà manifest không nói

**1. Prevalence tính trên dòng quan sát được, không phải trên tổng.** Một nhãn missing không
phải nhãn âm. Chia cho tổng số dòng sẽ làm prevalence thấp đi một cách giả tạo, và đó là lỗi
dễ lọt vào paper nhất.

**2. Event count quyết định protocol đánh giá.** `outcomes.csv` có cột
`recommended_protocol`: endpoint nào quá ít event thì hold-out đơn không đáng tin, phải dùng
repeated/nested CV (`source/data/cv.py`). Đây là quy tắc proposal đặt ra, không phải tuỳ chọn.

**3. Rule coverage là sàn của silver label.** `rule_coverage.csv` chạy đúng `apply_rule` thật
của `source/silver/rules.py` trên report thật. Target nào regex không giải quyết được thì hoặc
phải trả bằng LLM, hoặc thành abstention. Biết trước con số này thì biết arm `data.silver.rule`
sẽ đạt coverage bao nhiêu mà không cần chạy GPU.

## Khi thiếu thứ gì

Dataset chưa build → báo lỗi kèm lệnh build đúng profile, exit 3.
Bảng nào thiếu → ghi vào `inventory.csv` và bỏ qua phần đó, không crash.
Thiếu `matplotlib` → mọi phần số vẫn chạy; figure ghi `status: failed` kèm lý do.
`PE_CLOUD_ROOT` chưa set → exit 2, hoặc truyền `--dataset-root` / `--output-root`.
