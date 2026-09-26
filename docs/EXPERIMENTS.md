# Bảng kết quả cho paper

## Cách lấy số

Train rồi evaluate. `train_task.py` chỉ ghi history; metric test kèm CI chỉ có sau `evaluate.py`.

Số ở `result.json → evaluation.metrics.<tên> = {value, ci_low, ci_high}`.
Mỗi ô điền `value [ci_low–ci_high]`. Bootstrap 2000 lần, resample theo `patient_id`.

## Hai họ launcher, hai biến khác nhau

| Launcher | Biến chọn encoder | Giá trị |
|---|---|---|
| `probe.sh`, `global_single.sh`, `report_only.sh`, `anatomy_*.sh`, `*_student.sh` | `ENCODER_SOURCE` | `pretrained` `dapt` `c0` `silver` `diagnosis` `custom` |
| `single-task/*/global.sh`, `multi-task/global.sh`, `5_prognosis/<cohort>/<profile>/<task>/*` | `WEIGHT_SOURCE` | `pretrained` `dapt` `alignment` `rspect_multitask` `rspect_single` `silver_encoder` `custom` |

`alignment` = `c0`, `silver_encoder` = `silver`. Dùng nhầm biến thì launcher lấy mặc định và
4 arm ghi đè lên nhau.

Họ thứ hai tự đặt `experiment.id` / `output_id`, nên evaluate phải lặp lại đúng hai override
đó. Với `DATASET=full_inspect` thì `output_id` không có hậu tố dataset.

---

# BẢNG CHÍNH

## T1. Silver pretraining có cải thiện encoder không

```bash
# Probe — encoder đóng băng
for SRC in pretrained dapt c0 silver; do
  python run.py run probe.diag --set encoder.init_source=$SRC
  python tools/tasks/evaluate.py --config configs/runs/02_representation/probe/diagnosis.yaml --set encoder.init_source=$SRC --allow-full
done
```

```bash
# Fine-tune LoRA — cùng 4 stage
for SRC in pretrained dapt c0 silver; do
  python run.py run diag.global.single --set data.profile=full_inspect --set encoder.init_source=$SRC --gpus 0
  python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/baseline/global_single.yaml --set encoder.init_source=$SRC --set data.profile=full_inspect --allow-full
done
```

| Encoder stage | Probe auroc | Probe auprc | Fine-tune auroc | Fine-tune auprc | Fine-tune f1 |
|---|---|---|---|---|---|
| Public backbone | | | | | |
| + DAPT | | | | | |
| + image–report (C0) | | | | | |
| **+ silver (C_silver)** | | | | | |

## T2. Single-task hay multitask

Ba nhãn protocol (`pe_positive`, `pe_acute`, `pe_subsegmental`) chỉ có ở matrix launcher →
dùng `WEIGHT_SOURCE`.

```bash
# hàng 1 — single-task, C0
python tools/tasks/train_task.py --config configs/runs/03_diagnosis/matrix/single_task.yaml --set data.profile=full_inspect --set encoder.init_source=c0 --set experiment.id=DX_matrix_single_pe_positive_ds_full_inspect_weight_alignment_global --set experiment.output_id=single_task/pe_positive/weight_alignment/global --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/matrix/single_task.yaml --set data.profile=full_inspect --set encoder.init_source=c0 --set experiment.id=DX_matrix_single_pe_positive_ds_full_inspect_weight_alignment_global --set experiment.output_id=single_task/pe_positive/weight_alignment/global --allow-full
```

```bash
# hàng 2 — multitask, C0
python tools/tasks/train_task.py --config configs/runs/03_diagnosis/matrix/multitask.yaml --set data.profile=full_inspect --set encoder.init_source=c0 --set experiment.id=DX_matrix_multi_ds_full_inspect_weight_alignment_global --set experiment.output_id=multi_task/multi/weight_alignment/global --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/matrix/multitask.yaml --set data.profile=full_inspect --set encoder.init_source=c0 --set experiment.id=DX_matrix_multi_ds_full_inspect_weight_alignment_global --set experiment.output_id=multi_task/multi/weight_alignment/global --allow-full
```

```bash
# hàng 3 — single-task, C_silver
python tools/tasks/train_task.py --config configs/runs/03_diagnosis/matrix/single_task.yaml --set data.profile=full_inspect --set encoder.init_source=silver --set experiment.id=DX_matrix_single_pe_positive_ds_full_inspect_weight_silver_encoder_global --set experiment.output_id=single_task/pe_positive/weight_silver_encoder/global --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/matrix/single_task.yaml --set data.profile=full_inspect --set encoder.init_source=silver --set experiment.id=DX_matrix_single_pe_positive_ds_full_inspect_weight_silver_encoder_global --set experiment.output_id=single_task/pe_positive/weight_silver_encoder/global --allow-full
```

```bash
# hàng 4 — multitask, C_silver
python tools/tasks/train_task.py --config configs/runs/03_diagnosis/matrix/multitask.yaml --set data.profile=full_inspect --set encoder.init_source=silver --set experiment.id=DX_matrix_multi_ds_full_inspect_weight_silver_encoder_global --set experiment.output_id=multi_task/multi/weight_silver_encoder/global --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/matrix/multitask.yaml --set data.profile=full_inspect --set encoder.init_source=silver --set experiment.id=DX_matrix_multi_ds_full_inspect_weight_silver_encoder_global --set experiment.output_id=multi_task/multi/weight_silver_encoder/global --allow-full
```

Launcher in ra `experiment.id` và `output_id` đã resolve — đối chiếu trước khi evaluate.

| Mode | Init | auroc | auprc | sensitivity | specificity | f1 |
|---|---|---|---|---|---|---|
| single-task | C0 | | | | | |
| multitask | C0 | | | | | |
| single-task | C_silver | | | | | |
| multitask | C_silver | | | | | |

## T3. Anatomy và baseline

```bash
# hàng 1 — report-only
python run.py run diag.report_only --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/baseline/report_only.yaml --set data.profile=full_inspect --allow-full
```

```bash
# hàng 2 — global image
python run.py run diag.global.single --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/baseline/global_single.yaml --set data.profile=full_inspect --allow-full
```

```bash
# hàng 3 — global + silver head
python run.py run diag.global.silver_multitask --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/baseline/global_silver_multitask.yaml --set data.profile=full_inspect --allow-full
```

```bash
# hàng 4 — anatomy-aware
python run.py run diag.anatomy.concat --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/anatomy/single_concat.yaml --set data.profile=full_inspect --allow-full
```

```bash
# hàng 5 — anatomy + silver
python run.py run diag.anatomy.silver.concat --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/anatomy/silver_concat.yaml --set data.profile=full_inspect --allow-full
```

| Arm | Experiment | Ảnh | Mask | Silver | auroc | auprc | f1 |
|---|---|---|---|---|---|---|---|
| Report-only | `diag.report_only` | ✗ | ✗ | ✗ | | | |
| Global image | `diag.global.single` | ✓ | ✗ | ✗ | | | |
| Global + silver head | `diag.global.silver_multitask` | ✓ | ✗ | ✓ | | | |
| Anatomy-aware | `diag.anatomy.concat` | ✓ | ✓ | ✗ | | | |
| **Anatomy + silver** | `diag.anatomy.silver.concat` | ✓ | ✓ | ✓ | | | |

## T4. Prognosis — ảnh thêm gì so với clinical

Symlink dưới `scripts/5_prognosis/` bị checkout thành file text trên Windows → dùng dạng gọi
trực tiếp.

```bash
# hàng 1 — clinical
python tools/tasks/train_task.py --config configs/runs/04_prognosis/modality/ehr.yaml --set experiment.output_id=pe_positive_only/EHR_0_h/1_month_mortality/weight_alignment/clinical_only
python tools/tasks/evaluate.py --config configs/runs/04_prognosis/modality/ehr.yaml --set experiment.output_id=pe_positive_only/EHR_0_h/1_month_mortality/weight_alignment/clinical_only --allow-full
```

```bash
# hàng 2 — ảnh
python tools/tasks/train_task.py --config configs/runs/04_prognosis/modality/image.yaml --set experiment.output_id=pe_positive_only/EHR_0_h/1_month_mortality/weight_alignment/image_only
python tools/tasks/evaluate.py --config configs/runs/04_prognosis/modality/image.yaml --set experiment.output_id=pe_positive_only/EHR_0_h/1_month_mortality/weight_alignment/image_only --allow-full
```

```bash
# hàng 3 — ảnh + clinical
python tools/tasks/train_task.py --config configs/runs/04_prognosis/modality/image_ehr.yaml --set experiment.output_id=pe_positive_only/EHR_0_h/1_month_mortality/weight_alignment/image_clinical
python tools/tasks/evaluate.py --config configs/runs/04_prognosis/modality/image_ehr.yaml --set experiment.output_id=pe_positive_only/EHR_0_h/1_month_mortality/weight_alignment/image_clinical --allow-full
```

| Modality | Experiment | auroc | auprc | brier | calibration_slope |
|---|---|---|---|---|---|
| sPESI | `prog.spesi` | | | | |
| Clinical (EHR) | `prog.ehr` | | | | |
| Ảnh | `prog.image` | | | | |
| **Ảnh + clinical** | `prog.image_ehr` | | | | |

Câu trả lời: so `prog.ehr` với `prog.image_ehr`.

## T5. Vùng giải phẫu nào tác động tới PE

```bash
# bước 0 — model tham chiếu, bắt buộc trước
python run.py run diag.anatomy.concat --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/anatomy/single_concat.yaml --set data.profile=full_inspect --allow-full
```

```bash
# cột necessity — không cần evaluate riêng
python run.py run anatomy.remove_pa
python run.py run anatomy.remove_heart
python run.py run anatomy.remove_lung
python run.py run anatomy.remove_random
```

```bash
# cột sufficiency — paired bootstrap so với full model
# predictions.csv of the reference run (test rows are used); legacy predictions.parquet is also accepted
REF="$(ls -d $PE_CLOUD_ROOT/pe-project/outputs/diagnosis/DX_anatomy_concat*/epoch_*/predictions.csv | sort -V | tail -n 1)"
for R in pa heart lung random; do
  python run.py run anatomy.${R}_student --set data.profile=full_inspect --gpus 0
  python tools/tasks/evaluate.py --config configs/runs/05_anatomy_analysis/students/gt/$R.yaml --set data.profile=full_inspect --reference-predictions "$REF" --allow-full
done
```

| Vùng | Necessity: delta_probability | Sufficiency: student auroc | Δ vs full (paired) |
|---|---|---|---|
| Pulmonary artery | | | |
| Heart | | | |
| Lung parenchyma | | | |
| **Random control** | | | |

Đọc kết quả:

- necessity mạnh + sufficiency mạnh → vùng mang tín hiệu chính
- necessity mạnh + sufficiency yếu → cần nhưng không đủ một mình
- necessity yếu + sufficiency mạnh → thông tin lặp ở vùng khác
- cả hai yếu → không đóng góp

Random control bắt buộc: nếu xoá vùng ngẫu nhiên cũng tụt tương đương thì model chỉ phản ứng
với việc có vùng nào đó bị che, không phải với giải phẫu.

## T6. Vùng nào mang tín hiệu tiên lượng

Bản prognosis của T5. Email liệt kê ba arm này riêng: strict-heart, artery-only, lung-only.

```bash
# bước 0 — hàng trần, bắt buộc trước
python run.py run prog.image --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/04_prognosis/modality/image.yaml --set data.profile=full_inspect --allow-full
```

```bash
# bốn arm organ-only, train + evaluate
for R in heart pa lung random; do
  python run.py run prog.${R}_student --set data.profile=full_inspect --gpus 0
done
```

```bash
# Δ paired so với hàng trần
# predictions.csv of the reference run (test rows are used); legacy predictions.parquet is also accepted
REF="$(ls -d $PE_CLOUD_ROOT/pe-project/outputs/prognosis/PR_image_only*/epoch_*/predictions.csv | sort -V | tail -n 1)"
for R in heart pa lung random; do
  python tools/tasks/evaluate.py --config configs/runs/04_prognosis/students/$R.yaml --set data.profile=full_inspect --reference-predictions "$REF" --allow-full
done
```

| Vùng | Experiment | auroc | auprc | brier | calibration_slope | Δ auroc vs full CT (paired) |
|---|---|---|---|---|---|---|
| Toàn bộ CT | `prog.image` | | | | | — |
| Pulmonary artery | `prog.pa_student` | | | | | |
| Heart | `prog.heart_student` | | | | | |
| Lung parenchyma | `prog.lung_student` | | | | | |
| **Random control** | `prog.random_student` | | | | | |

Cả năm hàng cùng kiến trúc (`task.regions: []`, image-only), cùng init C0, cùng LoRA, cùng
cohort/split. Khác nhau duy nhất là voxel nào model được nhìn.

Đọc kết quả:

- PA cao → tiên lượng đến từ chính huyết khối
- Heart cao nhưng PA thấp → tiên lượng đến từ strain thất phải, không phải burden
- Lung cao → tiên lượng phần lớn là bệnh phổi nền, không phải PE
- không hàng nào vượt random → ảnh không mang tín hiệu tiên lượng riêng; dừng claim ở đây

Random control bắt buộc: nhãn tử vong tương quan với tuổi, bệnh nền và protocol chụp, những
thứ nhìn thấy được ở gần như mọi vùng cơ thể.

## T7. External validation chính (Turkey)

```bash
# bước 0 — segmentation/ROI bằng đúng contract của INSPECT
python run.py run data.segmentation.turkey --gpus 0 --allow-full
python run.py run data.roi.turkey --allow-full
```

```bash
# evaluate-only checkpoint diagnosis có head/architecture tương thích
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/external/turkey_test.yaml --checkpoint "$PE_CLOUD_ROOT/pe-project/outputs/diagnosis/DX_anatomy_concat/best.ckpt" --allow-full
```

| Checkpoint | auroc | auprc | sensitivity | specificity | Δ vs INSPECT test |
|---|---|---|---|---|---|
| C_diagnosis | | | | | |

Threshold chọn trên INSPECT validation, khóa trong source `result.json`, rồi áp một lần lên
Turkey test. `train_task.py` từ chối config này. C0/C_silver là encoder-only nên không được
strict-load như classifier hoàn chỉnh.

---

# PHỤ LỤC

## S1. Chất lượng silver label

```bash
python run.py run data.silver.medgemma --allow-full --gpus 0
```

Số có sẵn trong `outputs/silver_label/<method>/result.json → evaluation`. Đếm số, không CI.

| Cascade | accepted | abstained | no_result | coverage | disagreement_rate |
|---|---|---|---|---|---|
| `rule` | | | | | — |
| `falcon` | | | | | — |
| `medgemma` | | | | | — |

Báo coverage cạnh mọi metric dùng silver — precision cao do abstain nhiều không phải cải thiện.

## S2. Cascade nào tạo ra encoder tốt hơn

```bash
for M in medgemma; do
  python tools/tasks/train_task.py --config configs/runs/02_representation/silver_adaptation/$M.yaml --gpus 0
  python run.py run probe.diag --set encoder.init_source=custom --set encoder.checkpoint="$CKPT" --set encoder.source_experiment="SE_silver_${M}"
  python tools/tasks/evaluate.py --config configs/runs/02_representation/probe/diagnosis.yaml --set encoder.init_source=custom --set encoder.checkpoint="$CKPT" --set experiment.id=DX_probe_diagnosis_silver_${M} --allow-full
done
```

| Silver source | Probe auroc | Probe auprc | coverage của silver |
|---|---|---|---|
| `medgemma` | | | |
| `medgemma` | | | |
| `medgemma` | | | |
| C0 (không silver) | | | — |

## S3. Fusion

```bash
# concat + MLP
python run.py run diag.anatomy.silver.concat --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/anatomy/silver_concat.yaml --set data.profile=full_inspect --allow-full
```

```bash
# late-logit
python run.py run diag.anatomy.silver.late --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/anatomy/silver_late_logit.yaml --set data.profile=full_inspect --allow-full
```

```bash
# soft-MoE
python run.py run diag.anatomy.silver.moe --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/anatomy/silver_soft_moe.yaml --set data.profile=full_inspect --allow-full
```

| Fusion | Experiment | auroc | auprc | f1 |
|---|---|---|---|---|
| Concat + MLP | `diag.anatomy.silver.concat` | | | |
| Late-logit | `diag.anatomy.silver.late` | | | |
| Soft-MoE | `diag.anatomy.silver.moe` | | | |

## S4. Prognosis — encoder init

```bash
for WS in pretrained dapt alignment silver_encoder; do
  python tools/tasks/train_task.py --config configs/runs/04_prognosis/modality/image_ehr.yaml --set encoder.init_source=$WS --set experiment.output_id=pe_positive_only/EHR_0_h/1_month_mortality/weight_$WS/image_clinical
  python tools/tasks/evaluate.py --config configs/runs/04_prognosis/modality/image_ehr.yaml --set experiment.output_id=pe_positive_only/EHR_0_h/1_month_mortality/weight_$WS/image_clinical --allow-full
done
```

| Init | `WEIGHT_SOURCE` | auroc | auprc | brier | calibration_slope |
|---|---|---|---|---|---|
| Public | `pretrained` | | | | |
| DAPT | `dapt` | | | | |
| C0 | `alignment` | | | | |
| C_silver | `silver_encoder` | | | | |

## S5. Prognosis — cohort và endpoint

```bash
for CH in PE_positive all_patient; do
  for T in 1_month_mortality 12_month_mortality 12_month_PH; do
    python tools/tasks/train_task.py --config configs/runs/04_prognosis/modality/image_ehr.yaml --set data.cohort=$CH --set data.ehr_profile=EHR_0_h --set task.primary_target=$T --set experiment.output_id=$CH/EHR_0_h/$T/image_clinical
  done
done
```

| Cohort | Endpoint | n | positive rate | auroc | auprc |
|---|---|---|---|---|---|
| `pe_positive_only` | `1_month_mortality` | | | | |
| `pe_positive_only` | `12_month_mortality` | | | | |
| `pe_positive_only` | `12_month_PH` | | | | |
| `all_comers` | `1_month_mortality` | | | | |
| `all_comers` | `12_month_mortality` | | | | |

Censored/missing không phải negative. `n` và `positive rate` ở `result.json → data`.

## S6. QC artifact hỗ trợ

```bash
python run.py run data.segmentation --allow-full --gpus 0,1
python run.py run data.roi --allow-full
```

Đếm số, không CI.

| Artifact | pass | failed | ghi chú |
|---|---|---|---|
| 19 anatomy mask | | | cross-model lung dice: mean/median/min/max |
| ROI1..ROI7 | | | |
| ROI8 control (ROI2/4/6) | | | |

Kiểm tra ROI8 trong `roi_manifest.csv`:

| Kiểm tra | Cột | Kỳ vọng |
|---|---|---|
| Không chồng ROI nguồn | `overlap_voxels`, `dice_with_source` | 0 và 0.0 |
| Không chồng anatomy cấm | `forbidden_overlap_voxels` | 0 |
| Khớp thể tích vật lý | `physical_volume_error_mm3` | 0.0 |
| Tái lập được | `generation_seed` | cố định theo patient/study/ROI |
| Thất bại trung thực | `failure_reason` | `no_valid_control_location` |

---

# Không vào paper

**Contour** (`deferred.contour`): localize cục máu đông, câu hỏi khác. Thiếu
`manifests/contour.csv`, thiếu embolus mask, config trỏ checkpoint `DX01` không còn active.

**Knowledge distillation** (`scripts/4_diagnosis/*_student_kd.sh`): chỉ cần khi có claim riêng
về KD.

# Lưu ý

`seed` không được stamp vào run ID → nhiều seed cùng config sẽ va `OutputCollisionError`.
Với một seed, CI chỉ đo dao động lấy mẫu test, không đo dao động training.

Mọi arm trong cùng bảng dùng chung contract (`training.yaml#diagnosis` hoặc `#probe`).

Gần như mọi experiment đang `blocked` vì `configs/components/backbones.yaml` còn
`factory`/`output_adapter` rỗng và `feature_dim: 0`.

```bash
python run.py plan <experiment>
python run.py preflight <experiment>
```
