# 04. Experiments (diagnosis, prognosis, anatomy analysis, bảng paper)

**Đọc khi:** muốn biết mỗi nhóm experiment trả lời câu hỏi gì, chạy và đánh giá bằng lệnh nào, cái nào đang `blocked` và vì sao, và điền số vào bảng paper thế nào.

**Code chính:** `source/tasks/diagnosis/{model,losses,organ_targets,heads}.py`, `source/tasks/prognosis/{model,heads,losses}.py`, `source/engine/task_steps.py`, `source/clinical/{encoder,preprocessing,spesi}.py`, `source/data/build/{manifest_writer,adjudication}.py`, `tools/tasks/{zeroshot_penet,zeroshot_radar,score_baseline,counterfactual,evaluate}.py`, `source/components/roi/masks.py`, `source/roi/counterfactual.py`, `source/metrics/paired.py`, `configs/runs/**`, `configs/experiments.yaml`

Cơ chế train/evaluate dùng chung (launcher, Trainer, threshold, bootstrap, preview) ở [03_training_evaluation.md](03_training_evaluation.md); kiến trúc model ở [02_models.md](02_models.md); dữ liệu, mask, ROI, silver ở [01_data_pipeline.md](01_data_pipeline.md); thư mục output ở [05_running_outputs.md](05_running_outputs.md).

Trạng thái lấy từ `python run.py list` (trường `status` tĩnh của `configs/experiments.yaml`). Run ID là `experiment.id`; output ở `<outputs>/<family>/<RUN_ID>/epoch_<training.epochs>/`.

---

# 1. Diagnosis

**Câu hỏi:** từ một volume CTPA (hoặc feature CT-FM đã cache), dự đoán PE tốt đến đâu; FM + adaptation, mask giải phẫu, silver label, fusion và external validation đóng góp gì? **Diagnosis chỉ dùng ảnh**, không có EHR.

## 1.1 Nhãn, loss, target phụ

- **Nhãn native** lấy từ INSPECT khi build manifest (`manifests/diagnosis.csv`): `pe_positive_nlp` → `pe_present` (primary của hầu hết arm) và alias `pe_positive` (matrix, `DX_penet_style`); `pe_acute`; `pe_subsegmentalonly` → `pe_subsegmental_only` / `pe_subsegmental`. Giá trị rỗng thành NaN, `label_valid = isfinite`, loss và metric bỏ các dòng đó. `result.csv` của `pe_present` và `pe_positive` có giá trị `target` khác nhau khi ghép bảng.
- **Target phụ (silver)**: acuity, vị trí huyết khối, dấu hiệu thất phải, tràn dịch... không có nhãn native; là silver label do MedGemma trích từ báo cáo, chỉ dòng `status == accepted`. Silver chỉ là supervision phụ, **không bao giờ** là nhãn đánh giá.
- **Loss** (`task_loss_step` → `diagnosis` trong `task_steps.py`): `L = L_main` (BCE mỗi head binary, cross-entropy cho head nhiều lớp, chỉ trên dòng valid, nhân `task.loss_weights`); cộng `organ_auxiliary_loss` khi arm anatomy có head silver theo organ (mỗi organ một head trên feature của nhánh organ đó, trọng số organ preset 0.2); hoặc cộng `silver_loss_weight × diagnosis_loss(silver)` ở arm global silver. Không có class weight tự sinh.
- **Organ targets** (`DEFAULT_ORGAN_TARGET_MAPPING`): `heart` (rv_enlargement, rv_lv_ratio_*, septal_bowing, contrast_reflux, pericardial_effusion...), `pa` (acuity, central, lobar, segmental, subsegmental, saddle), `lung` (pleural_effusion, chronic_lung_disease, fibrosis, emphysema). Preset thật `tasks.yaml#diagnosis_organ_silver`: tất cả `source: silver`, `silver_loss_weight: 0.0`.
- **Target phụ không được chấm.** `evaluate.py` chỉ chấm primary + các head native binary khác; head silver, multiclass và head organ không có AUROC/threshold/CI. Thông tin duy nhất về chúng là loss và `valid_count` theo epoch trong `history.csv` (ví dụ `val_auxiliary.pa.acuity.loss`). Câu hỏi "silver có giúp không" trả lời bằng metric của primary target so với arm không silver.
- Mặc định (`training.yaml#diagnosis`): LoRA r8/α16 (CT-FM target `layers.3.blocks`, `layers.4.blocks`), `epochs: 50`, `batch_size: 1`, `gradient_accumulation: 4`, `learning_rate: 1e-4`, `early_stopping_patience: 10`, `threshold_method: youden`, `bootstrap_samples: 2000`, `seed: 42`.
- Manifest: `manifests/diagnosis.csv` (arm ảnh), `manifests/ct_fm/diagnosis.csv` (CT-FM frozen, trỏ tới feature cache). Split là official patient-level split của INSPECT. Profile `data.profile`: `smoke_30`, `test_500_sample`, `full_inspect` (mặc định).

## 1.2 Foundation và zero-shot (`configs/runs/01_foundation/`)

| Experiment | Run ID | Câu hỏi | Trạng thái |
|---|---|---|---|
| `foundation.ct_fm_frozen.diagnosis` | `DX_ctfm_frozen` | Representation CT-FM public đông lạnh mang bao nhiêu tín hiệu PE? | ready (cần feature cache) |
| `foundation.ct_fm_frozen.anatomy_concat.diagnosis` | `DX_ctfm_anatomy_concat` | Feature CT-FM pool theo organ có hơn vector global? | ready (cache + mask + ROI) |
| `foundation.ct_fm_frozen.anatomy_moe.diagnosis` | `DX_ctfm_anatomy_moe` | Router Soft-MoE có hơn concat trên feature organ? | ready |
| `diag.zeroshot.penet` | `DX_zeroshot_penet` | Model PE CTPA có sẵn đạt bao xa khi chưa adapt? | ready |
| `diag.zeroshot.radar` | `DX_zeroshot_radar` | Generalist vision-language CT có nhận ra PE không? | ready (cần env RADAR) |

CT-FM frozen ghi đè ngân sách: `epochs: 100`, `early_stopping_patience: 15`, `batch_size: 4`, `gradient_accumulation: 1`, `peft.method: frozen`. Arm global đọc `pooled_path` (`[513,1,1,1]`, `preload_inputs: true`); arm anatomy đọc grid đầy đủ vì cần pool theo mask.

```bash
PROFILE=full_inspect ACTION=all bash scripts/tool/run_ctfm_frozen.sh       # cache -> train -> evaluate, bỏ qua bước đã xong
PROFILE=smoke_30 EPOCHS=1 bash scripts/diagnosis/foundation/ctfm_frozen.sh
```

**Lưu ý CT-FM frozen:** nhánh global dùng embedding có trọng số body-coverage (`global_embedding`) nên ô đệm không khí của canvas cố định không vào vector global. `organ_adapter.standardize_inputs: true` z-score từng nhánh (fit chỉ trên train, lưu trong checkpoint, tóm tắt ở `result.json` → `model.feature_standardization`); checkpoint khác kiểu standardizer không load được (strict). Grad-CAM của CT-FM cached có độ phân giải thật bằng một ô feature (nội suy chỉ làm mượt), montage có thể gồm lát padding.

**Zero-shot** (không train, không có `epoch_<E>`, đọc NIfTI gốc, chỉ dự đoán validation + test; threshold Youden trên validation áp cho test; cùng định dạng `result.csv`/`predictions.csv`; bắt buộc `--allow-full` hoặc `--max-cases N`):

- **PENet:** trọng số release `third_party/weights/penet_best.pth.tar`; cửa sổ 32 lát không chồng lấn, xác suất series = `--aggregate max` (mặc định) hoặc `mean`; preview Grad-CAM cho `--preview-patients` (mặc định 5).
- **RADAR:** generalist CT bụng không có finding PE; chấm token organ động mạch phổi (`radar.organ: 肺动脉`) với cặp câu `radar.prompts` (đổi prompt là đổi arm). Cần env conda riêng (`transformers==4.25`): `zeroshot_radar.py` (env project) gọi `zeroshot_radar_worker.py` (env RADAR, `radar.python` hoặc `$RADAR_PYTHON`). Study không tìm thấy động mạch phổi bị bỏ, liệt kê ở `result.json` (`skipped_series`, `skipped_count`); điểm ghi từng study nên `--resume` được; preview không có CAM.

```bash
python tools/tasks/zeroshot_penet.py --config configs/runs/01_foundation/zero_shot/penet.yaml --allow-full --gpus 0
python run.py run diag.zeroshot.penet --allow-full --gpus 0
```

## 1.3 Global và anatomy (`configs/runs/02_diagnosis/{global,anatomy}/`)

Backbone `ct_fm` (image-space, LoRA), `encoder.init_source: pretrained`, nhãn `pe_present`.

| Experiment | Run ID | Câu hỏi | Supervision | Trạng thái |
|---|---|---|---|---|
| `diag.global.penet_style` | `DX_penet_style` | FM + adaptation có hơn 3D CNN không-FM tương đương? | `pe_positive`, khởi tạo ngẫu nhiên, `peft.method: full` | ready |
| `diag.global.single` | `DX_global_single` | Chỉ bằng chứng toàn volume dự đoán PE tốt đến đâu? (reference) | native | ready |
| `diag.global.silver_multitask` | `DX_global_silver_multitask` | Silver accepted có cải thiện head PE khi không có anatomy? | native + 13 head silver, `silver_loss_weight: 0.2` | ready (cần silver) |
| `diag.anatomy.concat` | `DX_anatomy_concat` | Pool vào nhánh tim/PA/phổi có hơn global-only? | native | ready (mask + ROI) |
| `diag.anatomy.silver.concat` | `DX_anatomy_silver_concat` | Supervise từng nhánh organ bằng silver có giúp? | native + head silver theo organ | ready |
| `diag.anatomy.silver.late` | `DX_anatomy_silver_late_logit` | Gộp *quyết định* nhánh có hơn gộp *feature*? | như trên, `late_logit` | ready |
| `diag.anatomy.silver.moe` | `DX_anatomy_silver_soft_moe` | Router học theo bệnh nhân có hơn concat và trung bình logit? | như trên, `soft_moe` | ready |

- `DX_anatomy_concat` dùng `tasks.yaml#diagnosis_anatomy_reference` (ROI `heart: ROI2`, `pa: ROI4`, `lung: ROI6` từ `roi/ROI_anatomy_and_controls/roi_manifest.csv`). Đây là model đông lạnh mà mọi counterfactual chấm (mục 3) và là nguồn của `encoder.init_source: diagnosis`; run ID và kiến trúc phải giữ ổn định.
- Cặp so sánh: `global.single` ↔ `global.silver_multitask` (silver, không anatomy); `global.single` ↔ `anatomy.concat` (anatomy); `anatomy.concat` ↔ `anatomy.silver.concat` (silver theo organ); ba arm silver với nhau (fusion).

```bash
python run.py plan diag.anatomy.concat ; python run.py preflight diag.anatomy.concat --gpus 0
python run.py run diag.anatomy.concat --gpus 0,1
python tools/launch.py --config configs/runs/02_diagnosis/anatomy/single_concat.yaml --gpus 0,1 --evaluate --allow-full
python run.py run diag.global.single --gpus 0 --set encoder.init_source=diagnosis     # run ID thêm __enc_diagnosis
```

## 1.4 Matrix (`configs/runs/02_diagnosis/matrix/`)

| Experiment | Run ID | Câu hỏi | Trạng thái |
|---|---|---|---|
| `diag.matrix.single_task` | `DX_matrix_single_task` | Mỗi nhãn PE đạt bao nhiêu khi không có task phụ? | ready |
| `diag.matrix.multitask` | `DX_matrix_multitask` | Train chung `pe_positive`/`pe_acute`/`pe_subsegmental` có giúp? | ready |

Cả hai: `finetuning.strategy: lora`, `variant_stamp: false`, không mask, không silver; multitask là ba head BCE cộng không trọng số, `pe_positive` là primary.

**Lưu ý:**
- Khối `diagnosis:` được `validate_config` kiểm: `mode` ∈ `single_task`/`multitask`, `tasks` ⊂ {`pe_positive`, `pe_acute`, `pe_subsegmental`} (single_task đúng 1, multitask đủ 3), `primary_task` ∈ `tasks`, `set(data.label_columns) == set(diagnosis.tasks)` và `task.primary_target == diagnosis.primary_task`. Đổi nhãn của `single_task` phải đổi **đồng bộ nhiều key** và `experiment.output_id` (vì `variant_stamp: false` khiến mọi nhãn ghi chung `DX_matrix_single_task/epoch_50/`). Comment config nhắc "shared matrix launcher" nhưng repo không có launcher đó:

```bash
python run.py run diag.matrix.single_task --gpus 0 \
  --set "diagnosis.tasks=[pe_acute]" --set diagnosis.primary_task=pe_acute \
  --set "data.label_columns=[pe_acute]" --set task.primary_target=pe_acute \
  --set "task.targets={pe_acute: 1}" --set experiment.output_id=DX_matrix_single_task/pe_acute
```
- Nhớ `evaluate.py` lặp lại đúng các `--set` này.

## 1.5 Baseline zoo (`configs/runs/02_diagnosis/baselines/{2D,2_5D,3D}/`)

20 arm `baseline.<model>`, run ID `DX_base_<model>`, đều ready, `task.architecture: baseline_classifier`, nhãn `pe_present`, official split.

| Nhóm | Model |
|---|---|
| 2D slice-MIL (32 lát axial, gated-attention) | `resnet18_2d`, `convnext_2d`, `vit_2d`, `swin_2d` |
| 2.5D (bộ 3 lát kề làm RGB) | `resnet18_25d`, `convnext_25d`, `vit_25d`, `swin_25d` |
| 3D | `resnet18_3d`, `resnet50_3d`, `densenet121_3d`, `convnext_3d`, `vit_3d`, `swin_3d`, `nnmamba_3d`, `mamba_mae_3d`, `vmamba_3d`, `penet_3d`, `ctfm_lora_3d`, `ctfm_frozen_3d` |

Ngân sách (`baselines.yaml`, chung cho mọi arm): tối đa 100 epoch, early stopping patience 15 theo val AUROC, `best.ckpt` = val AUROC cao nhất, batch hiệu dụng 4 (micro-batch × accumulation, micro-batch theo VRAM của từng arm), `compute.precision: auto`; `train_end_to_end` (full, cosine, lr 1e-4), `train_lora` (lr 3e-4), `train_frozen_features` (`ctfm_frozen_3d`, lr 1e-3). Bốn lưới (`tools/baselines/experiments.py`; một case = model × head × fraction × variant × seed, split chính thức):

| Lưới | File | Model | Head | Fraction train |
|---|---|---|---|---|
| exp01 | `scripts/diagnosis/baselines/exp01_baselines/experiment.yaml` | cả 20 | `mlp` | 100 |
| exp02 | `.../exp02_data_fraction/experiment.yaml` | 8 model 3D (`resnet18_3d`, `convnext_3d`, `vit_3d`, `swin_3d`, `nnmamba_3d`, `vmamba_3d`, `ctfm_lora_3d`, `ctfm_frozen_3d`) | `mlp` | 25, 50, 75, 100 |
| exp03 | `.../exp03_head_ablation/experiment.yaml` | 6 model 3D | `mlp`, `kan` | 100 |
| exp04 | `.../exp04_slice_ablation/experiment.yaml` | `resnet18_2d`, `resnet18_25d` × variant `default` (attention-MIL), `center` (lát giữa), `mean`, `max` | `mlp` | 100 |

Fraction chỉ subsample bệnh nhân **train** (phân tầng theo nhãn, lồng nhau, mỗi seed một bộ subset: `split_seed: per_seed`, xuất kèm kiểm tra ra `<task>/splits/data_fraction/seed_<s>/`); validation và official test không đổi. Seed lấy từ `--seeds` (mặc định `0 1 2`); mọi lưới chạy cho diagnosis và prognosis (`--task prognosis --label <outcome>`, `scripts/prognosis/baselines/`). Chi tiết: `scripts/diagnosis/baselines/README.md`, `scripts/README.md`.

```bash
python tools/baselines/run_case.py --model resnet18_3d --gpus 0                 # 1 case: prepare -> train -> evaluate
python tools/baselines/run_many.py --exp exp01_baselines --gpus 0,1,2,3         # cả lưới, bỏ qua case đã xong
GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh
```

## 1.6 External, chỉ test (`configs/runs/02_diagnosis/external/`)

| Experiment | Run ID | Checkpoint đánh giá | Threshold khoá từ | Trạng thái |
|---|---|---|---|---|
| `diag.external.turkey_test` | `DX_external_turkey_test` | `DX_anatomy_concat/epoch_50/checkpoint/best.ckpt` | `diagnosis/DX_anatomy_concat/epoch_50/result.json` | blocked: chưa nhận dữ liệu Turkey; cần `data.segmentation.turkey`, `data.roi.turkey` |

Hợp đồng `external_evaluation`: `test_only`, `prohibit_training`, `threshold_source: internal_validation_artifact`, `evaluation_split: test`. `train_task.py` từ chối config này; `evaluate.py` bắt buộc `--checkpoint`, chỉ chấm primary, threshold khoá theo SHA-256 checkpoint ([03](03_training_evaluation.md), mục threshold khoá).

```bash
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/external/turkey_test.yaml \
  --checkpoint <outputs>/diagnosis/DX_anatomy_concat/epoch_50/checkpoint/best.ckpt --allow-full --gpus 0
```

## 1.7 Lưu ý diagnosis (từ code review)

- `diagnosis_loss` bỏ qua head không có nhãn hợp lệ trong batch mà không thêm số hạng `×0` (khác `organ_auxiliary_loss`); dưới DDP có thể gây lỗi tham số không dùng (arm global silver, matrix multitask). Chưa kiểm chứng bằng chạy thật.
- Diagnosis không bỏ dòng thiếu nhãn (`drop_rows_without_labels` chỉ cho prognosis); với `batch_size: 1`, dòng thiếu nhãn primary sẽ raise `batch contains no valid diagnosis targets` (tuỳ manifest).
- `tasks.yaml#diagnosis_anatomy_reference` có vấn đề `_replace_` (xem mục 3.6).

---

# 2. Prognosis

**Câu hỏi:** từ CTPA index của bệnh nhân, dự đoán outcome nhị phân trong tương lai; ảnh thêm gì so với sPESI/EHR; fusion và anatomy giúp gì? Đây là **binary classification có mask**, không phải survival: cột time-to-event chỉ để audit.

## 2.1 Outcome, censoring, cohort

- **Bảy outcome native** (`DEFAULT_PROGNOSIS_OUTCOMES`, nghĩa all-cause): `1_month_mortality`, `6_month_mortality`, `12_month_mortality`, `1_month_readmission`, `6_month_readmission`, `12_month_readmission`, `12_month_PH`. Alias `mortality_30d` (= `1_month_mortality`) là `primary_target` của preset `tasks.yaml#prognosis`.
- **Censoring** (`adjudication.mortality_outcome`): mỗi outcome `X` sinh `X`, `X_status`, `X_observed`, `X_censored`, `X_time_to_event`, `X_source_censor_flag`. TRUE → 1, FALSE → 0, CENSORED/MISSING → **rỗng** (không bao giờ thành 0). Dataset đặt `NaN` + `label_valid`; prognosis bỏ hẳn dòng không có outcome nào quan sát được; loss (`masked_multitask_loss`) và evaluation chỉ tính phần tử valid của từng target (complete-case theo target).
- **Cohort** (index study = study sớm nhất của bệnh nhân thoả điều kiện; manifest do stage 0 viết):

| Manifest | Điều kiện | Tên | Dùng bởi |
|---|---|---|---|
| `manifests/prognosis.csv` | `pe_positive_nlp=TRUE` và `pe_acute=TRUE` | `confirmed_acute_pe` | mọi run `configs/runs/03_prognosis/**` |
| `manifests/prognosis_all_patient.csv` | không | `all_comers` | `foundation.ct_fm_frozen.prognosis_all_patient`, anatomy CT-FM frozen |
| `manifests/prognosis_pe_positive.csv` | `pe_positive_nlp=TRUE` | `pe_positive_only` | `foundation.ct_fm_frozen.prognosis_pe_positive` |

Split là split chính thức của INSPECT.

**Lưu ý:** `data.cohort` chỉ là **nhãn**: `evaluate.py` ghi nó vào `result.json` → `evaluation.cohort`, không code nào dùng nó để lọc. Cohort thực sự chọn bằng **`data.manifest`**; đổi cohort phải đổi cả hai.

## 2.2 Input và model

| Modality | Lấy từ | Encoder | Missing |
|---|---|---|---|
| `image` | `image_path` hoặc `pooled_path` | image encoder + `OrganAdapterBank` (`global` + `heart/pa/lung` nếu `task.regions` khác rỗng) | nhánh organ vắng khi không có mask |
| `ehr` | `data.ehr_columns` | `ClinicalEncoder` | NaN → impute + indicator |
| `spesi` | `data.spesi_columns: [spesi]` | `SpesiEncoder` (`[score, missing_flag]`) | NaN → 0 + cờ missing |

- **EHR:** `ClinicalPreprocessor` **chỉ fit trên split train** (lưu `center`, `scale`, `fill` làm buffer trong checkpoint); impute `mean`, z-score, nối missing mask (`ehr_include_missingness: true`) rồi MLP. Cửa sổ thời gian: profile `EHR_0_h` (trước giờ CTPA) và `EHR_24_h` (trước CTPA 24h), đều trước index; `data.ehr_profile` mặc định `EHR_0_h`.
- **sPESI:** 6 tiêu chí (tuổi > 80, cancer, cardiopulmonary disease, pulse ≥ 110, SBP < 100, SpO2 < 90), thiếu bất kỳ thành phần nào → không chấm. Stage 0 ghi `clinical/spesi_features.csv`, `spesi_components.csv`, `spesi_evaluable.csv`.
- **Model** (`PrognosisModel`): mọi branch đưa về cùng width, fusion **một lần ở cuối** (`concat_mlp`/`soft_moe`: nhánh vắng mask/renormalize; `late_logit`: mỗi branch có head riêng, trung bình logit có mask). Mỗi target một `PrognosisHeads`. Loss = `masked_multitask_loss(target_logits, ...)`, chỉ target có trong `data.label_columns`. `lineage.transfer_modules: [image_encoder]` khi khởi tạo từ checkpoint diagnosis.

## 2.3 Experiment prognosis (`configs/runs/03_prognosis/`)

Mọi run kế thừa `tasks.yaml#prognosis` + `prognosis_primary_cohort` (manifest `prognosis.csv`, target `mortality_30d`, trừ multitask).

| Experiment | Config | Modalities | Regions / fusion | Trạng thái |
|---|---|---|---|---|
| `prog.spesi` | `modality/spesi.yaml` (`PR_spesi_only`) | spesi | — | ready (chạy bằng `score_baseline.py`) |
| `prog.image` | `modality/image.yaml` (`PR_image_only`) | image | concat_mlp | ready |
| `prog.ehr` | `modality/ehr.yaml` | ehr | — | blocked |
| `prog.image_ehr` (= `prog.global.concat`) | `modality/image_ehr.yaml` | image, ehr | concat_mlp | blocked |
| `prog.global.late` / `prog.global.moe` | `global/{late_logit,soft_moe}.yaml` | image, ehr, spesi | late_logit / soft_moe | blocked |
| `prog.anatomy.concat` / `.late` / `.moe` | `anatomy/{concat,late_logit,soft_moe}.yaml` | image, ehr, spesi | heart, pa, lung | blocked |
| `prog.anatomy.multitask_moe` | `anatomy/multitask_soft_moe.yaml` | image, ehr, spesi | heart, pa, lung; soft_moe, 7 head | blocked |
| `ablation.ehr.{all,common}_{with,no}_missingness` | `ehr_ablation/*.yaml` | ehr | — | blocked |

`prog.anatomy.multitask_moe` dùng 7 outcome làm `label_columns`/`targets`, primary `1_month_mortality` (weight 1.0, 6 target phụ 0.2). EHR ablation có `experiment.stage: ablation`, `task.base_stage: prognosis`; biến thể `common` phải truyền `--set data.ehr_columns=[...]`. Prognosis **đang chạy được** khác: `foundation.ct_fm_frozen.{prognosis_all_patient,prognosis_pe_positive,anatomy_concat.prognosis,anatomy_moe.prognosis}` (image-only, CT-FM frozen, đủ 7 target; script `scripts/prognosis/foundation/ctfm_frozen_{all,pe}.sh`).

**Blocker EHR (preflight):** mọi arm có `ehr` fail vì (1) `ehr_feature_contract` cần `len(data.ehr_columns) == task.ehr_input_dim` và > 0, nhưng `data.ehr_columns: []` (cố ý để trống, "never guess") còn `task.ehr_input_dim: 32`; (2) `ehr_temporal_profile` yêu cầu `data.ehr_columns` bằng đúng `manifest_feature_columns` của profile. Stage 0 hiện chỉ sinh **7 cột utilization** (`ehr_index_time_missing`, `ehr_missing`, `ehr_has_crosswalk`, `ehr_events_prior`, `ehr_events_365d`, `ehr_events_30d`, `ehr_numeric_events_prior`), xung đột với 32. Muốn qua: điền 7 cột đó và đặt `ehr_input_dim: 7`, hoặc stage 0 phải sinh biến lâm sàng thật. `prog.spesi` không bị chặn.

## 2.4 Baseline sPESI: `score_baseline.py`

sPESI là điểm cố định, **không train** (train encoder trên một scalar có thể đảo thứ tự ca).

```bash
python tools/tasks/score_baseline.py --config configs/runs/03_prognosis/modality/spesi.yaml \
  --score-column spesi --allow-full \
  --restrict-to <derived>/cache/<profile>/clinical/spesi_evaluable.csv [--transform platt|minmax|raw_sigmoid]
```

Lấy dòng validation/test có score và nhãn hữu hạn; biến score thành xác suất bằng transform đơn điệu (`platt` mặc định, fit trên validation; `minmax`; `raw_sigmoid` chỉ có ý nghĩa về discrimination); threshold Youden trên validation; trên test tính `prognosis_metrics` + patient bootstrap + calibration curve. Output `<outputs>/prognosis/PR_spesi_only/score_baseline/` (`result.json` với `baseline.kind = "fixed_score"`, `trained: false`), không ghi vào `epoch_<E>/`. Để so công bằng, chấm arm ảnh trên cùng danh sách ca:

```bash
python tools/tasks/evaluate.py --config configs/runs/03_prognosis/modality/image.yaml \
  --restrict-to <derived>/cache/<profile>/clinical/spesi_evaluable.csv --allow-full
```

## 2.5 Chạy và đánh giá

```bash
python run.py plan prog.image ; python run.py preflight prog.image
python run.py run prog.image --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_prognosis/modality/image.yaml --allow-full
```

Đổi cohort/endpoint phải đổi đủ các key liên quan (như `tools/baselines/run_case.py`):

```bash
python tools/tasks/train_task.py --config configs/runs/03_prognosis/modality/image.yaml \
  --set data.manifest=manifests/prognosis_all_patient.csv --set data.cohort=all_comers \
  --set "data.label_columns=[12_month_mortality]" --set task.primary_target=12_month_mortality \
  --set "task.targets={12_month_mortality: 1}" \
  --set experiment.output_id=all_comers/12_month_mortality/image_only
```

Chỉ đặt `data.cohort` thì không đổi bệnh nhân; chỉ đặt `task.primary_target` mà không đổi `data.label_columns` thì evaluate lỗi `primary target ... is absent from data.label_columns`. Evaluation chấm mọi `task.targets` có trong `label_columns`, mỗi target một threshold Youden riêng trên validation (validation một lớp → 0.5), CI chỉ trên test; `result.json` có `targets.<target>` (threshold, metrics, calibration curve) và `split_metrics`. `evaluation.split_strategy` mặc định `patient_holdout`.

## 2.6 Lưu ý prognosis (từ code review)

- Các arm global fusion không chỉ khác fusion: `prog.image_ehr` dùng `[image, ehr]`, còn `prog.global.late`/`moe` dùng `[image, ehr, spesi]` dù comment nói "Identical except for fusion"; `configs/experiments.yaml` lại ghi `[image, ehr]` cho late/moe và anatomy. Comment `anatomy/late_logit.yaml` nói 5 branch, thực tế 6 (thêm spesi).
- `build_dataset` chỉ bỏ dòng không nhãn khi `experiment.stage == "prognosis"`; EHR ablation (stage `ablation`) giữ dòng toàn censored, với `batch_size: 1` có thể gây `batch contains no valid targets with positive weight` (chưa chạy thử).
- `prog.spesi` vẫn train được qua `run.py run`, nhưng baseline đúng nghĩa là `score_baseline.py`.
- Vòng lặp Sweeps trong `scripts/README.md` chỉ đặt `data.cohort` và `task.primary_target`; theo mục 2.5 không đổi cohort và sẽ lỗi ở evaluate.

---

# 3. Anatomy analysis (`configs/runs/04_anatomy_analysis/`)

**Câu hỏi:** model anatomy-aware có thật sự *dùng* bằng chứng từ tim / động mạch phổi / phổi không (counterfactual, không train), và mỗi expert / router đóng góp bao nhiêu (architecture ablation, có train)?

| Câu hỏi | Cơ chế | Train? | Config |
|---|---|---|---|
| Necessity: model *đã train* có cần vùng X? | Xoá vùng X trên cùng scan, chấm lại bằng model frozen, so theo cặp | Không | `counterfactual/remove_{heart,pa,lung,random}.yaml` |
| Architecture contribution: thêm expert/router có giúp? | Retrain từng biến thể từ cùng backbone public, chỉ đổi `task.regions` và `fusion` | Có | `architecture/{global_only,global_heart,global_pa,global_lung,full_moe,full_no_router}.yaml` |

## 3.1 Model tham chiếu `DX_anatomy_concat`

Mọi counterfactual chấm **một** checkpoint: `diag.anatomy.concat` (`DX_anatomy_concat/epoch_50/checkpoint/best.ckpt`): CT-FM image-space + LoRA, branch global + heart (ROI2) + pa (ROI4) + lung (ROI6), `concat_mlp`, nhãn `pe_present`. Kiến trúc/data contract dùng chung qua `tasks.yaml#diagnosis_anatomy_reference`; counterfactual load checkpoint **strict**, nên lệch kiến trúc là load fail chứ không âm thầm đổi khoa học.

## 3.2 Counterfactual inference (`tools/tasks/counterfactual.py`)

```text
build_task_model -> load_checkpoint(source_checkpoint, strict=True) -> đóng băng mọi tham số, eval()
VALIDATION (volume gốc)  -> threshold = Youden
TEST, mỗi batch:  p_orig = sigmoid(model(volume, masks));  volume' = remove_roi(volume, masks[region], local_mean)
                  p_cf = sigmoid(model(volume', masks));   delta = p_cf - p_orig      # masks GỐC, không re-segment
point metrics (original vs counterfactual, cùng threshold) + paired_patient_bootstrap
```

- Không train, không re-segment (`frozen_model: true`, `segmentation_rerun: false`, `original_masks_reused: true`). Model nhận **mask gốc** khi chấm volume đã xoá; branch global và các branch khác cũng đổi vì encoder nhìn cả volume, nên delta đo tác động lên toàn model.
- Threshold chọn trên validation gốc, áp nguyên cho cả original và counterfactual.
- Phải là encoder image-space: `input_counterfactual` bị chặn khi `model.cached_features` (nên reference dùng `backbones.yaml#ct_fm`, không phải `#ct_fm_features`).
- Xoá bằng `local_mean`: dãn mask 3 voxel (`max_pool3d`), thay vùng bằng mean cường độ của vỏ dãn; tránh tạo khối 0 như artifact. Các policy khác: `global_mean`, `zero`, `noise_matched`.

| Experiment | ID | Vùng xoá | Mask | Câu hỏi |
|---|---|---|---|---|
| `anatomy.remove_heart` | `CF_remove_heart` | tim | `heart` = ROI2 | Bằng chứng tim có cần? |
| `anatomy.remove_pa` | `CF_remove_pa` | cây động mạch phổi | `pa` = ROI4 | Model có dùng bằng chứng PA? (quan trọng nhất) |
| `anatomy.remove_lung` | `CF_remove_lung` | nhu mô phổi | `lung` = ROI6 | Bằng chứng nhu mô có cần? |
| `anatomy.remove_random` | `CF_remove_random` | vùng ngẫu nhiên khớp thể tích | `random` = ROI8 (`roi_control_for: {random: ROI4}`) | Xoá bất kỳ vùng cỡ đó làm điểm dịch bao nhiêu? |

Mask `random` là **ROI8 tính sẵn** bởi stage `data.roi` ([01_data_pipeline.md](01_data_pipeline.md)); sinh control lúc chạy bị cấm (`matched_random_mask` luôn raise, `counterfactual.py` raise nếu batch không có `masks["random"]`).

**Chạy** (launcher **bắt buộc** một scope; thiếu thì dừng với "counterfactual inference requires --patient-id, --max-cases, or --allow-full"):

```bash
python run.py run anatomy.remove_pa --max-cases 5 --gpus 0          # smoke, run id thêm __SMOKE_<hash>
for R in heart pa lung random; do python run.py run anatomy.remove_$R --allow-full --gpus 0; done
```

Cần `DX_anatomy_concat` đã train xong; kiểm trước bằng `python run.py plan anatomy.remove_pa`.

**Output** `outputs/counterfactual/CF_remove_<region>/`: `counterfactual_predictions.parquet` (original/counterfactual/delta probability, `roi_volume`, `roi_fraction_of_body`...), `paired_bootstrap_metrics.parquet` (reference/comparison/delta/CI), `result.json` (`evaluation.{threshold, delta_probability{mean,median,mean_absolute}, point_metrics, paired_counterfactual_vs_original}`).

**Đọc kết quả:** `delta_probability` âm = xoá vùng làm giảm xác suất PE (xem `mean`, `median`, `mean_absolute`). **Luôn đọc so với `CF_remove_random`**: nếu xoá vùng ngẫu nhiên cũng tụt tương đương, model chỉ phản ứng với "có vùng bị xoá". CI của delta metric không chứa 0 → thay đổi có ý nghĩa. Code không tự paired-test giữa hai arm (PA vs random); đọc `result.json` cạnh nhau. Necessity chứng minh model *dùng* thông tin vùng đó, không chứng minh cơ chế nhân quả sinh học.

## 3.3 Architecture ablation

Sáu biến thể cùng initialization (CT-FM public), split, optimizer (`training.yaml#diagnosis`) và nhãn; **chỉ đổi** `task.regions` và fusion. `experiment.stage: ablation`, output `outputs/ablation/architecture/AB_arch_<variant>/epoch_50/`.

| Experiment | `task.regions` | Fusion | Bỏ gì | Câu hỏi |
|---|---|---|---|---|
| `ablation.arch.global_only` | `[]` | `concat_mlp` (1 branch) | mọi expert; không cần mask | Organ expert có thêm gì? |
| `ablation.arch.global_heart` | `[heart]` | `concat_mlp` | PA, lung | Riêng expert tim đóng góp? |
| `ablation.arch.global_pa` | `[pa]` | `concat_mlp` | heart, lung | Riêng expert PA đóng góp? |
| `ablation.arch.global_lung` | `[lung]` | `concat_mlp` | heart, PA | Riêng expert phổi đóng góp? |
| `ablation.arch.full_moe` | `[heart, pa, lung]` | `soft_moe` | không | Đủ expert + router có hơn mọi cấu hình một phần? |
| `ablation.arch.full_no_router` | `[heart, pa, lung]` | `concat_mlp` | router | Router có thật sự làm việc? |

Cách so (cùng test patients): `global_<X>` vs `global_only` → đóng góp expert X; `full_no_router` vs `global_only` → cả bộ expert; `full_moe` vs `full_no_router` → router; `full_moe` vs `global_<X>` lẫn fusion và branch, đọc thận trọng.

```bash
for V in global_only global_heart global_pa global_lung full_moe full_no_router; do
  python run.py run ablation.arch.$V --set data.profile=full_inspect --gpus 0
done
python tools/tasks/evaluate.py --config configs/runs/04_anatomy_analysis/architecture/full_moe.yaml --allow-full \
  --reference-predictions <OUTPUT_ROOT>/ablation/architecture/AB_arch_full_no_router/epoch_50/predictions.csv
```

## 3.4 Trạng thái và lưu ý (từ code review)

Cả 10 entry `ready`. Bản local chưa có run `DX_anatomy_concat`, `CF_*`, `AB_arch_*`; thứ tự cần: `data.segmentation` → `data.roi` (ROI2/4/6/8 trên full cohort) → `diag.anatomy.concat` (train + evaluate) → bốn counterfactual; ablation chỉ cần ROI và chạy song song được.

- **ROI8 control cho ROI4 có thể không khả thi.** ROI8 khớp thể tích với ROI4 (PA) và bị cấm chồng anatomy; nếu không tìm được vị trí (`failure_reason = no_valid_control_location`) thì `anatomy.remove_random` có thể fail. Heart/lung lớn hơn nhiều, so với random không cùng thể tích; dùng `roi_volume`/`roi_fraction_of_body` để cân nhắc.
- **Vấn đề `_replace_`.** `tasks.yaml#diagnosis_anatomy_reference` ghi `task.targets: {_replace_: true, pe_present: 1}`, nhưng config resolve của `single_concat.yaml` và `remove_*.yaml` vẫn có đủ 14 target của `tasks.yaml#diagnosis` (`_load_with_bases` tiêu thụ `_replace_` khi merge preset vào `{}`, rồi mapping được merge vào 14 target kế thừa). Hệ quả: `DX_anatomy_concat`, `CF_*`, `AB_arch_*` đều resolve 14 head; loss chỉ tính `pe_present` (`data.label_columns`). Counterfactual vẫn load strict được vì hai phía resolve giống nhau.
- **`full_no_router` trùng `DX_anatomy_concat`** về kiến trúc và data contract (4 branch + concat), chỉ khác `experiment.id`/stage: là một lần retrain độc lập, hữu ích để thấy biến thiên giữa các lần train.
- `configs/experiments.yaml` liệt kê "ROI manifest" là requirement của `ablation.arch.global_only`, nhưng config đặt `data.roi_manifest: null`, `require_masks: false`.

---

# 4. Bảng kết quả cho paper

Train rồi evaluate: `train_task.py` chỉ ghi history; metric test kèm CI chỉ có sau `evaluate.py`. Số ở `result.json` → `evaluation.metrics.<tên> = {value, ci_low, ci_high}`; mỗi ô điền `value [ci_low–ci_high]` (bootstrap 2000 lần theo `patient_id`). Encoder chọn bằng `--set encoder.init_source=pretrained|diagnosis|custom`; mỗi override khác baseline được gắn vào tên run (ví dụ `__enc_diagnosis`), và evaluate phải lặp lại đúng các `--set` đã dùng khi train. Mọi arm trong cùng bảng dùng chung contract (`training.yaml#diagnosis` hoặc `#prognosis`). Kiểm trước: `python run.py plan <experiment>`, `python run.py preflight <experiment>`.

## T1. Single-task hay multitask

Dùng matrix (mục 1.4); đổi nhãn single-task cần override đủ key như ở đó.

```bash
python run.py run diag.matrix.single_task --gpus 0 --set data.profile=full_inspect
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/matrix/single_task.yaml --set data.profile=full_inspect --allow-full
python run.py run diag.matrix.multitask --gpus 0 --set data.profile=full_inspect
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/matrix/multitask.yaml --set data.profile=full_inspect --allow-full
```

| Mode | auroc | auprc | sensitivity | specificity | f1 |
|---|---|---|---|---|---|
| single-task | | | | | |
| multitask | | | | | |

## T2. Anatomy và baseline

```bash
for E in diag.global.single diag.global.silver_multitask diag.anatomy.concat diag.anatomy.silver.concat; do
  python run.py run $E --set data.profile=full_inspect --gpus 0
done
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/global/global_single.yaml --set data.profile=full_inspect --allow-full
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/global/global_silver_multitask.yaml --set data.profile=full_inspect --allow-full
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/anatomy/single_concat.yaml --set data.profile=full_inspect --allow-full
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/anatomy/silver_concat.yaml --set data.profile=full_inspect --allow-full
```

| Arm | Experiment | Ảnh | Mask | Silver | auroc | auprc | f1 |
|---|---|---|---|---|---|---|---|
| Global image | `diag.global.single` | ✓ | ✗ | ✗ | | | |
| Global + silver head | `diag.global.silver_multitask` | ✓ | ✗ | ✓ | | | |
| Anatomy-aware | `diag.anatomy.concat` | ✓ | ✓ | ✗ | | | |
| **Anatomy + silver** | `diag.anatomy.silver.concat` | ✓ | ✓ | ✓ | | | |

## T3. Prognosis: ảnh thêm gì so với clinical

`prog.ehr` và `prog.image_ehr` đang blocked (blocker EHR ở mục 2.3). `prog.spesi` chấm bằng `score_baseline.py` (mục 2.4).

```bash
python tools/tasks/score_baseline.py --config configs/runs/03_prognosis/modality/spesi.yaml --score-column spesi --allow-full
for M in ehr image image_ehr; do
  python tools/tasks/train_task.py --config configs/runs/03_prognosis/modality/$M.yaml
  python tools/tasks/evaluate.py --config configs/runs/03_prognosis/modality/$M.yaml --allow-full
done
```

| Modality | Experiment | auroc | auprc | brier | calibration_slope |
|---|---|---|---|---|---|
| sPESI | `prog.spesi` | | | | |
| Clinical (EHR) | `prog.ehr` | | | | |
| Ảnh | `prog.image` | | | | |
| **Ảnh + clinical** | `prog.image_ehr` | | | | |

Câu trả lời: so `prog.ehr` với `prog.image_ehr`. Chấm cả bốn trên cùng danh sách ca bằng `--restrict-to .../spesi_evaluable.csv`.

## T4. Vùng giải phẫu nào tác động tới PE

```bash
# bước 0: model tham chiếu (bắt buộc trước)
python run.py run diag.anatomy.concat --set data.profile=full_inspect --gpus 0
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/anatomy/single_concat.yaml --set data.profile=full_inspect --allow-full
# necessity: counterfactual cần --allow-full (hoặc --max-cases/--patient-id), không cần evaluate riêng
for R in pa heart lung random; do python run.py run anatomy.remove_$R --allow-full --gpus 0; done
```

| Vùng | Necessity: delta_probability |
|---|---|
| Pulmonary artery | |
| Heart | |
| Lung parenchyma | |
| **Random control** | |

Random control là bắt buộc để diễn giải (mục 3.2); xem lưu ý ROI8 ở mục 3.4.

## T5. External validation chính (Turkey)

Đang blocked (chưa có dữ liệu Turkey).

```bash
python run.py run data.segmentation.turkey --gpus 0 --allow-full
python run.py run data.roi.turkey --allow-full
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/external/turkey_test.yaml \
  --checkpoint "$PE_CLOUD_ROOT/pe-project/outputs/diagnosis/DX_anatomy_concat/epoch_50/checkpoint/best.ckpt" --allow-full
```

| Checkpoint | auroc | auprc | sensitivity | specificity | Δ vs INSPECT test |
|---|---|---|---|---|---|
| C_diagnosis | | | | | |

Threshold chọn trên INSPECT validation, khoá trong source `result.json`, áp một lần lên Turkey test.

## Phụ lục S1–S4

**S1. Chất lượng silver label** (`python run.py run data.silver.medgemma --allow-full --gpus 0`): số có sẵn ở `outputs/silver_label/medgemma/result.json` → `evaluation` (đếm số, không CI; chỉ còn method `medgemma`; mỗi dòng accepted có `source` là `medgemma` hoặc `rule`, đếm ở `by_source_and_status`, `disagreement_rate` từ `rule_medgemma_disagreement`). Báo coverage cạnh mọi metric dùng silver: precision cao do abstain nhiều không phải cải thiện.

| Source | accepted | abstained | no_result | coverage | disagreement_rate |
|---|---|---|---|---|---|
| `medgemma` | | | | | — |
| `rule` | | | | | — |

**S2. Fusion** (ba arm silver, `data.profile=full_inspect`):

```bash
for E in diag.anatomy.silver.concat diag.anatomy.silver.late diag.anatomy.silver.moe; do python run.py run $E --set data.profile=full_inspect --gpus 0; done
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/anatomy/silver_concat.yaml --set data.profile=full_inspect --allow-full   # + silver_late_logit.yaml, silver_soft_moe.yaml
```

| Fusion | Experiment | auroc | auprc | f1 |
|---|---|---|---|---|
| Concat + MLP | `diag.anatomy.silver.concat` | | | |
| Late-logit | `diag.anatomy.silver.late` | | | |
| Soft-MoE | `diag.anatomy.silver.moe` | | | |

**S3. Prognosis theo cohort và endpoint:** cohort chọn bằng `data.manifest` (mục 2.5), không chỉ `data.cohort`; manifest `prognosis_pe_positive.csv` (`pe_positive_only`) và `prognosis_all_patient.csv` (`all_comers`), endpoint `1_month_mortality`, `12_month_mortality`, `12_month_PH`. Dùng arm ảnh (`image.yaml`) vì arm có EHR đang blocked. Bảng: cohort × endpoint → n, positive rate, auroc, auprc. Censored/missing không phải negative; `n` và positive rate ở `result.json` → `data`.

**S4. QC artifact hỗ trợ** (`python run.py run data.segmentation --allow-full --gpus 0,1`, `python run.py run data.roi --allow-full`; đếm số, không CI): 19 anatomy mask (pass/failed, cross-model lung Dice mean/median/min/max), ROI1..ROI7, ROI8 control. Kiểm ROI8 trong `roi_manifest.csv`: `overlap_voxels` = 0 và `dice_with_source` = 0.0; `forbidden_overlap_voxels` = 0; `physical_volume_error_mm3` = 0.0; `generation_seed` cố định theo patient/study/ROI; thất bại trung thực ghi `failure_reason = no_valid_control_location`.

---

**Lưu ý chung:** `seed` không được stamp vào run ID nên nhiều seed cùng config va `OutputCollisionError`; một seed chỉ cho CI về dao động lấy mẫu test, không phải dao động training. Prognosis và external còn nhiều experiment `blocked`: dùng `python run.py plan <experiment>` để xem còn thiếu gì.
