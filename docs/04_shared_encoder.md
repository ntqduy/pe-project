# 04. Shared encoder và checkpoint

Chuỗi representation chính là `public CT backbone -> DAPT -> image-report alignment (C0)`.
RSPECT supervision và silver adaptation là nhánh so sánh, không được ngầm coi là C0.

## Module và mode train

- `model`: định nghĩa backbone/factory/feature adapter.
- `pretraining`: DAPT, train toàn encoder theo objective tự giám sát.
- `alignment`: image-report InfoNCE, train toàn encoder; text embedding đã precompute.
- `inference/evaluation`: load checkpoint, không cập nhật weight.
- diagnosis/prognosis: mặc định LoRA, optimizer chỉ nhận parameter `requires_grad=True`.
- representation probe: encoder frozen hoàn toàn (`requires_grad_(False)` + `.eval()` +
  `torch.no_grad()`), chỉ head probe được train. Probe **không** dùng LoRA:
  `source/engine/factory.py` return `RepresentationProbe` trước khi `apply_peft` được gọi,
  nên `peft.method` bị bỏ qua ở nhánh này.
  Đây là *linear evaluation protocol* (Alain & Bengio 2016; chuẩn dùng lại trong
  MoCo/SimCLR/DINO/MAE). Lưu ý: head diagnosis là một `nn.Linear`, nhưng head prognosis là
  `Linear → GELU → Linear` — tức **non-linear probe**, không so trực tiếp được với probe
  diagnosis.
  Head diagnosis đọc `task.targets`, nên probe chạy được cả hai chế độ: `probe.diag`
  (single-task, một head `pe_present`) và `probe.diag.multitask` (ba head độc lập
  `pe_positive` / `pe_acute` / `pe_subsegmental`, đúng ba nhãn native của
  `diag.matrix.multitask`). Config không khai `task.targets` thì mặc định vẫn là single-task
  `pe_present`.

`finetuning.strategy` hỗ trợ `full`, `linear_probe`, `lora`. `partial` hiện bị config
validator từ chối rõ ràng, vì code chưa định nghĩa layer nào thuộc partial.

## Thư mục `configs/runs/02_representation/probe`

Đây không phải weight hoặc dữ liệu. Ba config đóng băng encoder rồi train một head nhỏ, cố
định, để đo representation nào mang thông tin tốt hơn:

| config | experiment | head |
|---|---|---|
| `diagnosis.yaml` | `probe.diag` | 1 × `nn.Linear` → `pe_present` |
| `diagnosis_multitask.yaml` | `probe.diag.multitask` | 3 × `nn.Linear` → `pe_positive`, `pe_acute`, `pe_subsegmental` |
| `prognosis.yaml` | `probe.prog` | `Linear → GELU → Linear` (non-linear) |

Chúng dùng cùng optimizer/epoch/head/split contract cho mọi checkpoint; chỉ
`lineage.source_checkpoint` và run ID được đổi. Probe không fine-tune encoder và không thay
thế diagnosis/prognosis model cuối.

So sánh encoder stage thì chạy cả probe lẫn fine-tune và báo cáo cả hai — thứ hạng ở hai
nhánh có thể lệch nhau (Kornblith et al. 2019), và đó là kết quả đáng ghi chứ không phải lỗi:

```bash
for SRC in pretrained dapt c0 silver; do
  ENCODER_SOURCE=$SRC bash scripts/4_diagnosis/probe.sh             # single-task
  ENCODER_SOURCE=$SRC bash scripts/4_diagnosis/probe_multitask.sh   # multitask
done
```

`ENCODER_SOURCE` được stamp vào run ID (`__enc_dapt`, `__enc_silver`) nên các lần chạy không
đè nhau. Lưu ý `seed` **không** được stamp, nên chạy nhiều seed cho cùng một config sẽ va
`OutputCollisionError`; muốn multi-seed phải đặt tay `experiment.id` cho từng seed.

## Checkpoint contract

Mỗi `best.ckpt` chứa model state, optimizer/scheduler state và lineage. File sidecar
`best.ckpt.metadata.json` chứa tối thiểu:

`checkpoint_id` (SHA-256), `checkpoint_path`, `stage`, `source_checkpoint`, `dataset`,
`task`, `label_schema`, `architecture`, `finetuning_strategy`, `created_at`, `git_commit`,
`config_path`.

Khi load, report ghi `missing_keys`, `unexpected_keys`, `shape_mismatches`, số key/layer
encoder đã load, module được chọn và head có reset hay không. Shape mismatch trong strict
mode gây lỗi rõ ràng. Transfer `modules=[image_encoder]` cố ý giữ head downstream mới.

Nguồn init hợp lệ: `pretrained`, `dapt`, `c0`, `rspect_multitask`, `rspect_single`,
`silver`, `diagnosis`, `custom`. Aliases cũ vẫn được resolve trong migration.

Output training chuẩn:

```text
<run>/
  best.ckpt
  best.ckpt.metadata.json
  resolved_config.yaml
  lineage.json
  environment.json
  metrics.json
  result.json
  logs/run.log
  logs/history.csv
```

`history.csv` chứa loss/metric theo epoch. `run.log` chứa command, cấu hình PEFT, số
parameter total/trainable/frozen/LoRA, checkpoint load report, từng epoch và kết thúc/lỗi.
