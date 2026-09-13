# Cấu trúc config

Đọc từ `configs/runs/` trước. Chỉ mở catalog component khi cần xem chi tiết preset dùng
chung; catalog không chạy trực tiếp.

```text
configs/
  runs/
    00_data/             dataset, segmentation, ROI, silver
    01_foundation/       public encoder baselines
    02_representation/   DAPT, alignment, probe, RSPECT, silver adaptation
    03_diagnosis/        diagnosis runs và external test-only
    04_prognosis/        modality/fusion/anatomy runs
    05_anatomy_analysis/ counterfactual và ROI students
    06_deferred/         ngoài active protocol
  components/
    backbones.yaml       CT backbone contract
    encoders.yaml        nguồn checkpoint/init
    objectives.yaml      DAPT và alignment objectives
    tasks.yaml           task/cohort/supervision
    training.yaml        optimizer, PEFT, evaluation
    fusions.yaml         concat, late-logit, Soft-MoE
    anatomy.yaml         organ adapter và ROI source
    silver.yaml          bảy silver methods
  clinical/              PESI/sPESI contract
  compute/               CPU/GPU presets
  experiments.yaml       semantic registry cho run.py
  paths.yaml             data/output roots
```

## Cách đọc `_base_`

`../../../components/training.yaml#diagnosis` nghĩa là lấy preset `diagnosis` trong catalog
training. Base merge từ trên xuống; mapping merge đệ quy, list/scalar bị thay thế.
`_replace_: true` thay toàn bộ mapping, `_delete_` xóa key inherited. Setting trong run file
luôn thắng.

Quy tắc:

1. Chỉ file dưới `runs/` là runnable.
2. Run chỉ inherit root/compute/component, không inherit run khác.
3. Setting dùng chung mới vào catalog; setting một experiment ở ngay run file.
4. `experiment.name` là CLI key; `experiment.id` là output/checkpoint key và phải unique.
5. Không điền giả factory, weight, clinical column hoặc external manifest còn thiếu.

## `runs/02_representation/probe` để làm gì?

Đây là protocol đo chất lượng representation, không tạo encoder mới. `diagnosis.yaml` và
`prognosis.yaml` đóng băng encoder, train cùng một head nhỏ/cùng hyperparameter cho mọi
checkpoint. Chỉ source checkpoint và output ID được đổi; nếu đổi head/epoch/split thì kết quả
probe không còn so sánh trực tiếp.

## Các trục chính

- `data.profile`: `smoke_30`, `test_500_sample`, `full_inspect`.
- `model.backbone`: `ct_fm`, `ct_clip`, `totalfm`.
- `encoder.init_source`: `pretrained`, `dapt`, `c0`, `rspect_multitask`,
  `rspect_single`, `silver`, `diagnosis`, `custom`.
- `finetuning.strategy`: `full`, `linear_probe`, `lora`; `partial` chưa hỗ trợ.

Kiểm tra config qua:

```bash
python run.py show <experiment.name>
python run.py plan <experiment.name>
python run.py preflight <experiment.name>
```
