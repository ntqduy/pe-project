# 04. Experiments (baseline exp01-exp04, zero-shot, data stages, bảng paper)

**Đọc khi:** muốn biết mỗi nhóm experiment trả lời câu hỏi gì, chạy và đánh giá bằng lệnh nào, cái nào đang `blocked` và vì sao, và điền số vào bảng paper thế nào.

**Code chính:** `tools/baselines/{experiments,run_case,run_many,summarize,fraction_subsets}.py`, `scripts/tool/run_baseline_grid.sh`, `scripts/diagnosis/baselines/exp0*_*/experiment.yaml`, `source/engine/{factory,task_steps}.py`, `source/model/classifier.py`, `source/data/experiment_splits.py`, `source/data/build/{manifest_writer,adjudication}.py`, `tools/tasks/{zeroshot_penet,zeroshot_radar,evaluate}.py`, `source/metrics/paired.py`, `configs/runs/**`, `configs/experiments.yaml`

Cơ chế train/evaluate dùng chung (launcher, Trainer, threshold, bootstrap, preview) ở [03_training_evaluation.md](03_training_evaluation.md); kiến trúc model ở [02_models.md](02_models.md); dữ liệu, mask, ROI, silver ở [01_data_pipeline.md](01_data_pipeline.md); thư mục output ở [05_running_outputs.md](05_running_outputs.md). Hướng dẫn chạy đầy đủ của lưới baseline: [scripts/diagnosis/baselines/README.md](../scripts/diagnosis/baselines/README.md).

`configs/experiments.yaml` có 30 entry (`data.dataset.*`, `data.*`, `diag.zeroshot.*`, `baseline.<model>`); trạng thái lấy từ `python run.py list` (trường `status` tĩnh). Mọi run dùng **split chính thức** của INSPECT (train / validation / test theo bệnh nhân), lặp lại bằng seed; không có k-fold.

---

# 1. Data stages (`configs/runs/00_data/`)

| Experiment | Config | Sinh ra | Trạng thái |
|---|---|---|---|
| `data.dataset.{smoke_30,test_500_sample,full_inspect}` | `dataset/*.yaml` | manifest, audit, cache volume ([01](01_data_pipeline.md)) | ready |
| `data.segmentation` / `data.roi` | `segmentation/inspect.yaml`, `roi/inspect.yaml` | mask TotalSegmentator + QC, ROI1..ROI8 | ready |
| `data.segmentation.turkey` / `data.roi.turkey` | `segmentation/turkey.yaml`, `roi/turkey.yaml` | như trên cho Turkey | blocked (chưa có dữ liệu) |
| `data.silver.medgemma` | `silver/medgemma.yaml` | silver label từ báo cáo | ready |

Mask/ROI/silver là artifact dữ liệu (QC, phân tích); model baseline không đọc chúng.

---

# 2. Diagnosis

**Câu hỏi:** từ một volume CTPA (hoặc feature CT-FM đã cache), các encoder 2D / 2.5D / 3D phát hiện PE tốt đến đâu; dữ liệu train, loại head, cách nhìn lát cắt ảnh hưởng thế nào? **Chỉ dùng ảnh.**

- **Nhãn:** `pe_present` (từ `pe_positive_nlp` của INSPECT khi build `manifests/diagnosis.csv`); rỗng → NaN, `label_valid = isfinite`, loss và metric bỏ các dòng đó. CT-FM frozen đọc `manifests/ct_fm/diagnosis.csv` (trỏ tới feature cache).
- **Loss:** BCE-with-logits trên dòng valid (`diagnosis_loss`, `source/tasks/diagnosis/losses.py`), không class weight tự sinh.
- Profile `data.profile`: `smoke_30`, `test_500_sample`, `full_inspect` (mặc định).

## 2.1 Zero-shot (`configs/runs/01_foundation/zero_shot/`)

| Experiment | Run ID | Câu hỏi | Trạng thái |
|---|---|---|---|
| `diag.zeroshot.penet` | `DX_zeroshot_penet` | Model PE CTPA có sẵn đạt bao xa khi chưa adapt? | ready |
| `diag.zeroshot.radar` | `DX_zeroshot_radar` | Generalist vision-language CT có nhận ra PE không? | ready (cần env RADAR) |

Không train, không có `epoch_<E>`, đọc NIfTI gốc, chỉ dự đoán validation + test; threshold Youden trên validation áp cho test; cùng định dạng `result.csv`/`predictions.csv`; bắt buộc `--allow-full` hoặc `--max-cases N`. Runner riêng: `tools/tasks/zeroshot_penet.py`, `tools/tasks/zeroshot_radar.py`.

- **PENet:** trọng số release `third_party/weights/penet_best.pth.tar`; cửa sổ 32 lát không chồng lấn, xác suất series = `--aggregate max` (mặc định) hoặc `mean`; preview Grad-CAM cho `--preview-patients` (mặc định 5).
- **RADAR:** generalist CT bụng không có finding PE; chấm token organ động mạch phổi (`radar.organ: 肺动脉`) với cặp câu `radar.prompts` (đổi prompt là đổi arm). Cần env conda riêng (`transformers==4.25`): `zeroshot_radar.py` (env project) gọi `zeroshot_radar_worker.py` (env RADAR, `radar.python` hoặc `$RADAR_PYTHON`). Study không tìm thấy động mạch phổi bị bỏ, liệt kê ở `result.json` (`skipped_series`, `skipped_count`); điểm ghi từng study nên `--resume` được; preview không có CAM.

```bash
python tools/tasks/zeroshot_penet.py --config configs/runs/01_foundation/zero_shot/penet.yaml --allow-full --gpus 0
python run.py run diag.zeroshot.penet --allow-full --gpus 0
bash scripts/diagnosis/zero_shot/radar.sh
```

## 2.2 Baseline zoo (`configs/runs/02_diagnosis/baselines/{2D,2_5D,3D}/`)

20 arm `baseline.<model>`, đều ready, `task.architecture: baseline_classifier` (encoder → projection chung → head MLP/KAN, [02](02_models.md) §6).

| Nhóm | Model |
|---|---|
| 2D slice-MIL (32 lát axial, gated-attention) | `resnet18_2d`, `convnext_2d`, `vit_2d`, `swin_2d` |
| 2.5D (bộ 3 lát kề làm RGB) | `resnet18_25d`, `convnext_25d`, `vit_25d`, `swin_25d` |
| 3D | `resnet18_3d`, `resnet50_3d`, `densenet121_3d`, `convnext_3d`, `vit_3d`, `swin_3d`, `nnmamba_3d`, `mamba_mae_3d`, `vmamba_3d`, `penet_3d`, `ctfm_lora_3d`, `ctfm_frozen_3d` |

Ngân sách (`baselines.yaml`, chung cho mọi arm): tối đa 100 epoch, early stopping patience 15 theo val AUROC, `best.ckpt` = val AUROC cao nhất, batch hiệu dụng 4 (micro-batch × accumulation, micro-batch theo VRAM của từng arm), `compute.precision: auto`, DataLoader có seed; `train_end_to_end` (full, cosine, lr 1e-4), `train_lora` (lr 3e-4), `train_frozen_features` (`ctfm_frozen_3d`, lr 1e-3). Threshold Youden trên validation; CI bootstrap 2000 lần theo bệnh nhân trên test.

**CT-FM frozen** chỉ còn là arm `ctfm_frozen_3d` (feature CT-FM pooled đã cache + head). Cache dựng một lần mỗi profile, trước mọi case `ctfm_frozen_3d`:

```bash
bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh                     # full_inspect, cuda:0 (bọc tools/data/build_ctfm_cache.py)
PROFILE=smoke_30 GPUS='' bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh   # CPU; VERIFY_CACHE=1 / REBUILD_CACHE=1 / WORKERS
bash scripts/diagnosis/baselines/exp01_baselines/3D/ctfm_frozen_3d.sh
```

Output cũ `DX_ctfm_frozen` / `PR_ctfm_frozen_*` (nếu còn trên bucket) do đường code đã gỡ sinh ra, chỉ có giá trị lịch sử.

## 2.3 Bốn lưới thí nghiệm

Một case = model × head × fraction × variant × seed trên split chính thức (`tools/baselines/experiments.py` đọc `experiment.yaml`). Một run được định danh bằng setting, không bằng experiment, nên case trùng (vd. `resnet18_3d` + MLP + 100%) chỉ train một lần và dùng chung.

| Lưới | Script trong `scripts/diagnosis/baselines/` | Model | Head | Fraction train | Câu hỏi |
|---|---|---|---|---|---|
| exp01 | `exp01_baselines/{2D,2_5D,3D}/<model>.sh` | cả 20 | `mlp` | 100 | Encoder nào tốt nhất? |
| exp02 | `exp02_data_fraction/frac025/`, `frac050/`, `frac075/`, `frac100/` — mỗi thư mục `run_all.sh` + `<model>.sh` | 8 model 3D (`resnet18_3d`, `convnext_3d`, `vit_3d`, `swin_3d`, `nnmamba_3d`, `vmamba_3d`, `ctfm_lora_3d`, `ctfm_frozen_3d`) | `mlp` | 25, 50, 75, 100 | Cần bao nhiêu dữ liệu train? |
| exp03 | `exp03_head_ablation/<model>_kan.sh` | 6 model 3D (`resnet18_3d`, `convnext_3d`, `vit_3d`, `nnmamba_3d`, `ctfm_frozen_3d`, `ctfm_lora_3d`) | script chạy `kan`; nhánh `mlp` là run exp01 | 100 | KAN có hơn MLP trên cùng projection? |
| exp04 | `exp04_slice_ablation/{2D,2_5D}/<model>.sh` | `resnet18_2d`, `resnet18_25d` × variant `default` (32 lát, attention-MIL), `center` (lát giữa), `mean`, `max` | `mlp` | 100 | Nhìn cả volume quan trọng thế nào với 2D/2.5D? |

- **exp02:** chỉ subsample bệnh nhân **train** (phân tầng theo nhãn, lồng nhau 25 ⊂ 50 ⊂ 75 ⊂ 100); validation và official test không đổi. `split_seed: per_seed`: mỗi seed một bộ subset, xuất kèm `check.json` ra `<task>/splits/data_fraction/seed_<s>/` (`tools/baselines/fraction_subsets.py`; FAIL dừng case trước khi train).
- **exp02:** mỗi thư mục `frac0NN/` cố định một mức dữ liệu (`--fractions NN`), cùng tên với hậu tố `__frac0NN` của thư mục run; `frac100/` chính là các run exp01 (đã xong thì bỏ qua). `exp02_data_fraction/run_all.sh` chạy cả 4 mức.
- **exp03:** script chỉ chạy head KAN (`--heads kan`); nhánh MLP là run exp01 của cùng model (dùng chung), nên chạy exp01 trước rồi exp03. `summarize.py --exp exp03_head_ablation` ghép MLP và KAN thành bảng + `mlp_vs_kan.png`.
- **exp04:** variant là override trên run config, run tag `__v<variant>`; `default` chính là run exp01.
- Seed lấy từ `--seeds` (mặc định `0 1 2`).

```bash
python tools/baselines/run_case.py --model resnet18_3d --gpus 0                 # 1 case: prepare -> train -> evaluate
python tools/baselines/run_many.py --exp exp01_baselines --gpus 0,1,2,3         # cả lưới, bỏ qua case đã xong
GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp02_data_fraction/run_all.sh         # cả 4 mức dữ liệu
bash scripts/diagnosis/baselines/exp02_data_fraction/frac025/run_all.sh --gpus 0      # chỉ 25%
bash scripts/diagnosis/baselines/exp03_head_ablation/run_all.sh --gpus 0              # KAN (sau exp01)
bash scripts/tool/run_baseline_grid.sh exp04_slice_ablation all --variants "center mean" --dry-run
python tools/baselines/smoke_pipeline.py                                        # toàn pipeline trên dữ liệu tổng hợp (CPU)
```

Bảng tổng hợp mỗi lưới (`tools/baselines/summarize.py`, chạy sau mỗi grid): `summary.md`, `summary_pretty.csv`, `summary_ensemble.csv` (xác suất trung bình qua seed, ghép theo `study_id`), `summary.csv`, `summary_raw.csv` + plot. Mỗi dòng có thêm **Params (M)**, **Trainable (M)** và **GFLOPs / volume** (`ctfm_frozen_3d` ghi "(head only)" vì feature CT-FM đã tính trước).

---

# 3. Prognosis (image-only)

**Câu hỏi:** từ CTPA index của bệnh nhân, ảnh dự đoán outcome nhị phân tương lai tốt đến đâu? Cùng bốn lưới như diagnosis, `--task prognosis --label <outcome>` (hoặc `scripts/prognosis/baselines/exp0*.sh`). Đây là **binary classification có mask**, không phải survival.

- **Outcome** (`--label`, bắt buộc, mỗi lần gọi một outcome): `1_month_mortality`, `6_month_mortality`, `12_month_mortality`, `1_month_readmission`, `6_month_readmission`, `12_month_readmission`, `12_month_PH`.
- **Censoring** (`adjudication.mortality_outcome`): TRUE → 1, FALSE → 0, CENSORED/MISSING → **rỗng** (không bao giờ thành 0); dataset bỏ dòng không có outcome quan sát được, loss và evaluation chỉ tính phần tử valid.
- **Cohort** (`--cohort`): `all` (mặc định, `manifests/prognosis_all_patient.csv`, `all_comers`) hoặc `pe` (`manifests/prognosis_pe_positive.csv`, `pe_positive_only`); `run_case.py` đổi đồng bộ `data.manifest`, `data.cohort`, `data.label_columns`, `task.primary_target`, `task.targets`. CT-FM frozen dùng bản `manifests/ct_fm/` tương ứng.

```bash
bash scripts/prognosis/baselines/exp01_baselines.sh --label 12_month_PH --gpus 0 --seeds "0 1 2"
bash scripts/prognosis/baselines/exp02_data_fraction.sh --label 1_month_mortality --cohort pe
```

Output: `<outputs>/prognosis/BASE/<profile>/prognosis_<cohort>_<label>/{runs,exp0*_*}/`.

---

# 4. Bảng kết quả cho paper

Số lấy từ `summary_pretty.csv` / `summary.md` của từng lưới: AUROC / AUPRC dạng `value [ci_low–ci_high]` (seed ensemble, bootstrap 2000 lần theo `patient_id`) và mean ± std qua seed. Một run đơn: `result.json` → `evaluation.metrics.<tên> = {value, ci_low, ci_high}`.

| Bảng | Nguồn |
|---|---|
| T1. Baseline 2D / 2.5D / 3D (diagnosis) | `exp01_baselines/summary_pretty.csv` |
| T2. Kích thước tập train | `exp02_data_fraction/summary_pretty.csv`, `auroc_vs_fraction.png` |
| T3. MLP vs KAN | `exp03_head_ablation/summary_pretty.csv`, `mlp_vs_kan.png` |
| T4. Slice ablation | `exp04_slice_ablation/summary_pretty.csv`, `slice_ablation.png` |
| T5. Zero-shot | `DX_zeroshot_{penet,radar}/result.json` |
| T6. Prognosis theo outcome × cohort | các lưới prognosis, cùng định dạng |

So sánh có cặp hai model trên cùng bệnh nhân test: `python tools/tasks/evaluate.py --config <cfg> --allow-full --reference-predictions <run khác>/predictions.csv` (paired patient bootstrap, `source/metrics/paired.py`).

## Phụ lục

**S1. Chất lượng silver label** (`python run.py run data.silver.medgemma --allow-full --gpus 0`): số có sẵn ở `outputs/silver_label/medgemma/result.json` → `evaluation` (đếm số, không CI; chỉ còn method `medgemma`; mỗi dòng accepted có `source` là `medgemma` hoặc `rule`, đếm ở `by_source_and_status`, `disagreement_rate` từ `rule_medgemma_disagreement`).

| Source | accepted | abstained | no_result | coverage | disagreement_rate |
|---|---|---|---|---|---|
| `medgemma` | | | | | — |
| `rule` | | | | | — |

**S2. QC artifact hỗ trợ** (`python run.py run data.segmentation --allow-full --gpus 0,1`, `python run.py run data.roi --allow-full`; đếm số, không CI): 19 anatomy mask (pass/failed, cross-model lung Dice mean/median/min/max), ROI1..ROI7, ROI8 control. Kiểm ROI8 trong `roi_manifest.csv`: `overlap_voxels` = 0 và `dice_with_source` = 0.0; `forbidden_overlap_voxels` = 0; `physical_volume_error_mm3` = 0.0; `generation_seed` cố định theo patient/study/ROI; thất bại trung thực ghi `failure_reason = no_valid_control_location`.
