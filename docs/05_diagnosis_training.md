# 05. Diagnosis

## Nhãn native

Protocol matrix dùng đúng ba nhãn INSPECT đã review:

- `pe_positive` <- raw `pe_positive_nlp`;
- `pe_acute` <- raw `pe_acute`;
- `pe_subsegmental` <- raw `pe_subsegmentalonly`.

Giá trị missing/censored được để rỗng và mask khỏi loss, không đổi thành 0. Các target
report-derived khác chỉ là silver auxiliary supervision.

## Mode và loss

`configs/runs/03_diagnosis/matrix/single_task.yaml` có `diagnosis.mode=single_task` và
một task được chọn. `multitask.yaml` yêu cầu đúng cả ba task, với `pe_positive` là primary.
Config validator kiểm tra `diagnosis.tasks`, `data.label_columns` và model primary head
khớp nhau.

Mỗi head binary dùng BCE-with-logits trên các row observed. Loss multitask là tổng có
trọng số; hiện không bịa class weights. Nếu cần imbalance weighting, phải tính chỉ từ train
split, lưu vào config/result và chạy lại mọi arm so sánh.

## Fine-tuning và experiment tối thiểu

Diagnosis mặc định là LoRA trên module `attn` và `projection`, rank 8, alpha 16, dropout
0.05. Năm so sánh tối thiểu dùng cùng split/preprocessing/head/optimizer:

1. single-task từ published/pretrained;
2. single-task từ C0;
3. single-task từ C_silver;
4. multitask từ C0;
5. multitask từ C_silver.

DAPT và diagnosis checkpoint là các initialization bổ sung được hỗ trợ, không thay năm arm
chính dùng để trả lời câu hỏi silver supervision.

Mỗi run ID phải khác và metadata checkpoint ghi source checkpoint/hash. Không so metric của
run có split hoặc cohort khác.

### Early stopping

`training.early_stopping_patience: 10` (mặc định cho mọi contract trong
`configs/components/training.yaml`): dừng sau 10 epoch liên tiếp không cải thiện metric
validation. Đặt `null` để chạy hết `training.epochs` như trước.

- `best.ckpt` luôn là epoch tốt nhất, không phải epoch cuối — dừng sớm không mất model.
- Quyết định dừng tính từ giá trị đã reduce qua mọi rank, nên DDP không bị lệch ở barrier.
- `result.json` ghi `epochs` (ngân sách cấu hình), `epochs_run` (thực chạy),
  `stopped_early` và `early_stopping_patience`.
- So sánh các encoder stage vẫn hợp lệ vì patience giống nhau cho mọi arm. **Không** chỉnh
  patience riêng cho từng checkpoint — làm vậy là confound đúng biến đang muốn đo.

## Đánh giá

Threshold chỉ chọn trên validation rồi khóa cho test. Report gồm AUROC, AUPRC,
sensitivity, specificity, F1 và patient-level bootstrap CI; predictions nằm trong
`predictions.parquet`. Validation/test chỉ dùng native label.

```bash
python run.py preflight diag.matrix.single_task
python run.py run diag.matrix.single_task --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/matrix/single_task.yaml --allow-full
```

### Chọn encoder stage bằng `ENCODER_SOURCE`

Đây là biến quyết định cả chương trình so sánh: nó chọn checkpoint mà encoder khởi tạo từ đó,
và được stamp vào run ID (`__enc_dapt`, `__enc_silver`) nên các lần chạy không đè nhau.

```bash
for SRC in pretrained dapt c0 silver; do
  # fine-tune, single-task
  DATASET=full_inspect ENCODER_SOURCE=$SRC GPUS=0 bash scripts/4_diagnosis/single-task/pe_positive/global.sh
  # probe, single-task và 3 nhãn
  ENCODER_SOURCE=$SRC bash scripts/4_diagnosis/probe.sh
  ENCODER_SOURCE=$SRC bash scripts/4_diagnosis/probe_multitask.sh
done

# fine-tune, 3 nhãn
for SRC in c0 silver; do
  DATASET=full_inspect ENCODER_SOURCE=$SRC GPUS=0 bash scripts/4_diagnosis/multi-task/global.sh
done
```

Giá trị hợp lệ: `pretrained` (alias `published` / `public` / `original`), `dapt`,
`c0` (alias `image_report` / `alignment`), `silver`, `diagnosis`, `rspect_multitask`,
`rspect_single`, `custom`. Với `custom` phải kèm `ENCODER_CHECKPOINT` và
`ENCODER_EXPERIMENT` để provenance vẫn được ghi vào lineage.

Chạy cả probe lẫn fine-tune rồi báo cáo cả hai: thứ hạng ở hai nhánh có thể lệch nhau, và
đó là kết quả đáng ghi. Kế hoạch experiment đầy đủ: [EXPERIMENTS.md](EXPERIMENTS.md).

Silver input mới là `silver_labels.csv`, chỉ row `label_status=accepted`/legacy
`status=accepted` được dùng.
