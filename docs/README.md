# Tài liệu `docs/`

**Đọc khi:** mới vào repo, hoặc cần nhớ "cái gì nối với cái gì" trước khi đọc chi tiết từng stage.

**Code chính:** `run.py`, `configs/experiments.yaml`, `tools/launch.py`, `source/utils/config.py`, `source/engine/experiment.py`

Mỗi file trong `docs/` giải thích **code làm gì và tại sao**. Danh sách experiment chính thức luôn là `python run.py list`.

## 1. Repo trả lời câu hỏi gì

Framework nghiên cứu **pulmonary embolism (PE) trên CTPA 3D** của bộ Stanford INSPECT. Mỗi experiment trong `configs/experiments.yaml` có trường `question:`; gộp lại:

| Nhóm | Câu hỏi | Experiment tiêu biểu |
|---|---|---|
| Data | Cohort nào, mask/ROI/silver label nào dùng được? | `data.dataset.*`, `data.segmentation`, `data.roi`, `data.silver.medgemma` |
| Zero-shot | PENet, RADAR làm được tới đâu khi không train? | `diag.zeroshot.*` |
| Baseline (diagnosis) | Backbone 2D / 2.5D / 3D public (gồm CT-FM LoRA / frozen) phát hiện PE end-to-end tốt tới đâu? Cần bao nhiêu dữ liệu train, MLP hay KAN, nhìn bao nhiêu lát? | `baseline.*`, lưới exp01-exp04 |
| Baseline (prognosis) | Cùng các lưới, ảnh dự đoán mortality / readmission / PH tốt tới đâu? | `scripts/prognosis/baselines/exp0*.sh --label <outcome>` |

Bảng kết quả cần điền cho paper: [04_experiments.md](04_experiments.md).

## 2. Pipeline end-to-end

```text
 Stanford INSPECT (raw, READ-ONLY)  CT + report + EHR(MEDS/OMOP)
        |
        v
 [0] DATASET  data.dataset.{smoke_30,test_500_sample,full_inspect}
        |     eligibility, CT integrity, adjudication nhãn, giữ official split,
        |     manifest, cache CT, EHR, sPESI
        |
        +--------------------------+------------------------------+
        v                          v                              v
 [1] SEGMENTATION           [2] SILVER LABELS              [3] ZERO-SHOT
     data.segmentation          data.silver.medgemma           diag.zeroshot.{penet,radar}
     TotalSegmentator +         MedGemma đọc report,
     LungMask Dice QC           accepted / abstained
        |
        v
 [1b] ROI  data.roi
     ROI1..ROI8 + random control

 [4] BASELINE  baseline.* / exp01-exp04 (scripts/diagnosis/baselines/)
     diagnosis (PE +/-) và prognosis image-only (--task prognosis --label <outcome>)
     ctfm_frozen_3d cần cache: scripts/diagnosis/baselines/prepare_ctfm_cache.sh
```

- Stage 0 là điều kiện của mọi thứ. Segmentation / ROI / silver là artifact dữ liệu (QC, phân tích); baseline không cần mask.
- Baseline load thẳng public weight (`encoder.init_source=pretrained`).
- **Không stage nào tự chạy prerequisite.** `python run.py plan <experiment>` chỉ báo cái còn thiếu.

## 3. `run.py` → registry → config → launch → tasks

`run.py` không chứa logic khoa học: nó dịch **tên ngữ nghĩa** (`baseline.resnet18_3d`) thành một file config rồi giao cho đúng CLI.

```text
python run.py run baseline.resnet18_3d --gpus 0,1 --set data.profile=test_500_sample
   |  load_registry()        đọc configs/experiments.yaml -> {config, status, requires, ...}
   |  experiment_stage()     load_config(entry.config) -> experiment.stage
   v
 delegate(mode=run)
   |-- status=unavailable                -> từ chối
   |-- entry có runner:                  -> tools/tasks/zeroshot_penet.py | zeroshot_radar.py
   |-- stage dataset|segmentation|roi    -> tools/data/build_dataset.py
   |                                        tools/create_masks/generate_masks.py
   |                                        tools/build_rois/build_rois.py
   '-- còn lại                           -> tools/launch.py --config <file> [--set ...] [--gpus ...]
                                             |  resolve config, require_preflight()
                                             |    silver         -> tools/silver_labels/generate_silver_labels.py
                                             |    diagnosis/prognosis -> tools/tasks/train_task.py
                                             |    --evaluate     -> tools/tasks/evaluate.py
                                             v
                                  0-1 GPU = python, >=2 GPU = torchrun (DDP)
```

| Lệnh | Làm gì |
|---|---|
| `list` / `show <name>` | In danh sách experiment kèm status / mô tả, config, `experiment.id`, requires, blockers |
| `plan <name>` | Đi ngược chuỗi `requires`, đánh dấu READY / MISSING / BLOCKED |
| `preflight <name>` | Kiểm tra path, manifest, weight, checkpoint, output collision (không train) |
| `dry <name>` | In lệnh `tools/launch.py` sẽ chạy |
| `run <name>` | Chạy thật |

- Data stage, zero-shot, silver bắt buộc phạm vi tường minh (`--patient-id`, `--max-cases` / `--max-reports`, hoặc `--allow-full`). Training (diagnosis/prognosis) **không** nhận cờ scope: phạm vi chính là dataset profile.
- `run.py run baseline.*` chỉ **train** một run; metric test có sau bước evaluate ([03_training_evaluation.md](03_training_evaluation.md)). Lưới exp01-exp04 (prepare → train → evaluate, nhiều seed) chạy qua `tools/baselines/run_case.py` / `run_many.py`.

## 4. Hệ thống config

Mỗi run config trong `configs/runs/` ghép từ các preset dùng chung qua `load_config()` (`source/utils/config.py`):

```text
load_config(path, overrides)
  1. _load_with_bases()      đọc _base_: [file.yaml, catalog.yaml#preset, ...]; deep_merge đệ quy;
                             _replace_: true thay hẳn mapping, _delete_: [...] xoá key kế thừa
  2. apply_overrides()       --set a.b.c=VALUE (VALUE parse bằng YAML: 1, true, [a,b])
  3. expand_environment()    ${PE_CLOUD_ROOT} ... -> biến môi trường
  4. validate_config()       resolve_backbone, resolve_encoder_initialization,
                             stamp_experiment_variant, kiểm tra prefix id theo stage,
                             config_hash = SHA-256 config khoa học (bỏ resume/overwrite)
```

Run config **không bao giờ** kế thừa từ run config khác; cách đọc `_base_` chi tiết: [`configs/README.md`](../configs/README.md).

**`experiment.id` → thư mục output** (`source/engine/experiment.py`):

```text
<output_root>/<FAMILY_PATHS[experiment.family]>/<experiment.output_id or experiment.id>[/epoch_<training.epochs>]
ví dụ  .../outputs/diagnosis/DX_base_resnet18_3d/epoch_100/
```

`OutputManager.prepare()` từ chối ghi đè thư mục đã có (`OutputCollisionError`) trừ khi `--resume` / `--overwrite`. Chi tiết: [05_running_outputs.md](05_running_outputs.md).

**Run-id stamping** (`stamp_experiment_variant()`): giá trị lệch khỏi baseline của config được gắn vào id, nên các tổ hợp không đè nhau:

```text
DX_base_resnet18_3d                            full_inspect, backbone và init mặc định
DX_base_resnet18_3d__ds_test_500_sample        đổi data.profile
DX_base_resnet18_3d__bb_resnet50_3d            đổi model.backbone
```

- Hậu tố: `__ds_` (profile ≠ `experiment.baseline_dataset`), `__bb_` (backbone), `__enc_` (init source).
- `experiment.variant_stamp: false` thì không stamp (silver `medgemma.yaml`).
- `seed` **không** được stamp vào id: chạy nhiều seed qua `run.py` sẽ va collision. Lưới baseline (`run_case.py`) tự đặt mỗi seed vào `.../official_seed<S>/` ([05_running_outputs.md](05_running_outputs.md)).

## 5. Bốn trục độc lập

Mỗi run là một điểm trong không gian bốn chiều; tất cả là giá trị config, không có code path riêng, nên các arm so sánh được.

| Trục | Đặt bằng | Giá trị | Ảnh hưởng |
|---|---|---|---|
| Dataset profile | `--set data.profile=...` (wrapper: `PROFILE=`) | `smoke_30`, `test_500_sample`, `full_inspect` | Đọc `derived/datasets/<profile>/`; stamp `__ds_` |
| Method | Experiment chọn | Một run config mỗi arm | Encoder, PEFT, head, task |
| Encoder weight | `model.backbone` × `encoder.init_source` | `ct_fm` / zoo × `pretrained` (hoặc `--scratch`) | Stamp `__bb_`, `__enc_`; ghi vào `lineage` |
| Run scope | `--patient-id`, `--max-cases`, `--max-reports`, `--allow-full` | Chỉ cho generation stage | Smoke ghi thư mục riêng |

Ba profile dùng chung một contract (`source/data/profiles/_common.yaml`), chỉ khác khối sampling: chạy `test_500_sample` là diễn tập đúng code path của `full_inspect`.

## 6. Quy tắc thiết kế

- **Official split không đổi.** Build fail nếu một patient nằm ở hai split; training fraction của baseline zoo (exp02) chỉ cắt train; không có k-fold, lặp lại bằng seed.
- **Threshold và checkpoint chỉ chọn trên validation**, áp nguyên cho test.
- **Fail loudly.** Contract chưa có thì entry `blocked`; pretrained weight khớp quá ít tensor thì báo lỗi; sPESI thiếu thành phần thì để trống.
- **Không mock model**; kiểm tra bằng `preflight`, `dry`, `plan` và chạy một patient.
- **Silver label không bao giờ là nhãn đánh giá** (chỉ dòng `accepted`). Mọi model trainable chỉ dùng ảnh.
- Đổi weight bằng config, không fork file model. Output không nằm trong source tree.

## 7. Trạng thái hiện tại

Theo `python run.py list`: **30 experiment, 28 `ready`, 2 `blocked`**. `ready` chỉ nghĩa là config đầy đủ; artifact (manifest, mask, checkpoint) có thể chưa có trên máy: `plan` sẽ báo MISSING.

Nguyên nhân blocked duy nhất: **External cohort** (`data.segmentation.turkey`, `data.roi.turkey`): dữ liệu Turkey chuẩn hoá chưa được giao.

Proposal có nhắc nhưng **không có trong code**:

- **Chưa implement:** fine-tune nnU-Net riêng cho CTPA và annotation do chuyên gia review. Mask giải phẫu hiện là pseudo-label TotalSegmentator + LungMask.
- **Ngoài phạm vi / đã gỡ:** model anatomy-aware (pool theo vùng, fusion), counterfactual xoá vùng, prognosis đa modality (EHR / sPESI), staged pretraining, Falcon và LLM adjudication cho silver.
- **Chưa có kết quả:** chưa có run train/evaluate thật trên `full_inspect`, nên chưa có AUROC/CI điền [04_experiments.md](04_experiments.md).

## 8. Bản đồ thư mục

| Thư mục / file | Vai trò |
|---|---|
| `run.py` | Entrypoint người dùng: `list / show / plan / preflight / dry / run` |
| `configs/` | `experiments.yaml` (registry), `runs/` (một file mỗi experiment: `00_data`, `01_foundation/zero_shot`, `02_diagnosis/baselines`), `components/` (preset), `compute/`, `clinical/`, `paths.yaml` |
| `source/` | Thư viện, toàn bộ logic khoa học: `data/`, `clinical/`, `segmentation/`, `roi/`, `silver/`, `components/`, `tasks/`, `model/`, `engine/`, `metrics/`, `distributed/`, `imaging/`, `profiling/`, `utils/` |
| `tools/` | CLI Python theo domain (`data/`, `create_masks/`, `build_rois/`, `silver_labels/`, `tasks/`, `baselines/`) + `launch.py`, `preflight.py`, `run_status.py` |
| `scripts/` | Wrapper shell mỏng: biến môi trường → `run.py` / `tools/` ([`scripts/README.md`](../scripts/README.md)) |
| `analysis/` | EDA trên dataset profile đã build ([`analysis/README.md`](../analysis/README.md)) |
| `third_party/` | Repo upstream và weight cục bộ ([`third_party/README.md`](../third_party/README.md)) |
| `paper/` | PDF tham khảo (INSPECT, email yêu cầu, ...) |
| `output/` | Bản sao output cục bộ, cùng layout như `/mnt/pe-project/outputs` |

Bản đồ code chi tiết: [06_code_map.md](06_code_map.md).

## 9. Danh mục tài liệu

| File | Giải thích gì | Đọc khi |
|---|---|---|
| [01_data_pipeline.md](01_data_pipeline.md) | Stage 0 (dataset profile), segmentation, ROI, silver label | Trước khi build dataset / mask / ROI / silver |
| [02_models.md](02_models.md) | Kiến trúc: `BaselineClassifier`, encoder 2D/2.5D/3D, CT-FM, head MLP/KAN, PEFT | Khi đọc `source/components`, `source/model` |
| [03_training_evaluation.md](03_training_evaluation.md) | Train, evaluate, threshold, bootstrap, checkpoint | Trước khi train/evaluate bất kỳ arm nào |
| [04_experiments.md](04_experiments.md) | Data stages, zero-shot, lưới baseline exp01-exp04 (diagnosis + prognosis) và bảng paper | Khi chạy/đọc `baseline.*`, `diag.zeroshot.*`, hoặc chuẩn bị số cho paper |
| [05_running_outputs.md](05_running_outputs.md) | Chạy thực tế (script, env var, multi-GPU) và output mỗi stage | Khi ngồi vào máy chạy, hoặc mở một thư mục output |
| [06_code_map.md](06_code_map.md) | Call chain và file quan trọng theo package | Khi cần tìm/sửa code |

Thứ tự đọc cho người mới: README này → [05_running_outputs.md](05_running_outputs.md) (chạy thử `smoke_30` song song) → 01 → 02 → 03 → 04 → 06 khi cần sửa code.

Các README khác: [`README.md`](../README.md) (gốc, tiếng Anh), [`configs/README.md`](../configs/README.md), [`scripts/README.md`](../scripts/README.md), [`scripts/diagnosis/baselines/README.md`](../scripts/diagnosis/baselines/README.md), [`tools/README.md`](../tools/README.md), [`analysis/README.md`](../analysis/README.md), [`third_party/README.md`](../third_party/README.md).
