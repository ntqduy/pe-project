# Cấu trúc config

Đọc từ `configs/runs/` trước. Chỉ mở catalog component khi cần xem chi tiết preset dùng
chung; catalog không chạy trực tiếp.

```text
configs/
  runs/
    00_data/                 stage 0, chuẩn bị dữ liệu
      dataset/               smoke_30, test_500_sample, full_inspect
      segmentation/          inspect.yaml, turkey.yaml (TotalSegmentator + lungmask)
      roi/                   inspect.yaml, turkey.yaml (ROI1..ROI8 + random control)
      silver/                medgemma.yaml
    01_foundation/           backbone public, không học thêm encoder
      ct_fm_frozen/          CT-FM frozen: diagnosis, prognosis_all, prognosis_pe
                             (cũng là nơi build cache feature cho baseline ctfm_frozen_3d)
      zero_shot/             penet.yaml, radar.yaml
    02_diagnosis/
      baselines/             baseline zoo theo chiều: 2D/, 2_5D/, 3D/ (20 model); dùng cho cả
                             diagnosis và prognosis (run_case.py --task prognosis --label ...)
  components/
    backbones.yaml       CT backbone contract (registry của mọi encoder)
    encoders.yaml        nguồn checkpoint/init
    baselines.yaml       head/task/budget dùng chung cho baseline zoo
    tasks.yaml           task/cohort/supervision
    training.yaml        optimizer, PEFT, evaluation
    fusions.yaml         concat_mlp (head của CT-FM frozen)
    silver.yaml          medgemma silver method
  clinical/              sPESI source contract
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

## Các trục chính

- `data.profile`: `smoke_30`, `test_500_sample`, `full_inspect`.
- `model.backbone`: `ct_fm`, `ct_fm_features`, `penet_style`, and the baseline zoo in
  `components/backbones.yaml` (`resnet18_3d`, `vit_2d`, ... see `scripts/diagnosis/baselines/README.md`).
- `encoder.init_source`: `pretrained`, `diagnosis`, `custom`.
- `finetuning.strategy`: `full`, `linear_probe`, `lora`; `partial` chưa hỗ trợ.

Kiểm tra config qua:

```bash
python run.py show <experiment.name>
python run.py plan <experiment.name>
python run.py preflight <experiment.name>
```
