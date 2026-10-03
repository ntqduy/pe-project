# 06. Bản đồ code

**Đọc khi:** mở một file `.py` lạ và muốn biết nó thuộc stage nào, ai gọi nó; hoặc cần lần theo một lệnh chạy (`run.py`, `scripts/**/*.sh`) xuống tới code khoa học.

**Code chính:** `source/`, `tools/`, `analysis/`, `run.py`

Chỉ mô tả code của project (không gồm `third_party/`, `output/`). Bảng liệt kê file quan trọng của từng package; file nhỏ/hiển nhiên và `__init__.py` được bỏ qua.

## 1. Cách đọc nhanh

Ba tầng, gọi theo một chiều:

```text
scripts/**/*.sh        wrapper shell: đặt biến môi trường, chọn profile/GPU, rồi gọi Python
  -> run.py            "menu" theo tên experiment (configs/experiments.yaml)
    -> tools/**        CLI: argparse, preflight, ghi output, chia GPU/rank (không có logic khoa học)
      -> source/**     thư viện: dữ liệu, mô hình, huấn luyện, metric (không có CLI)
analysis/              EDA độc lập, chỉ đọc dataset đã build
```

- `source/model/2D_model/` và `source/model/3D_model/` bắt đầu bằng chữ số nên **không import trực tiếp được**: chỉ nạp lười qua `importlib` trong `source/model/registry.py`.
- Một số encoder/provider nạp bằng chuỗi `"module:function"` trong config (`configs/components/backbones.yaml`, `silver.yaml`), ví dụ `source.components.encoders.image.ct_fm:build_ct_fm_backbone`.

## 2. Luồng gọi chính

`->` là "gọi", `=>` là "chạy tiến trình con". Dispatch của `run.py` xem [README.md](README.md#3-runpy--registry--config--launch--tasks).

### 2.1 Train diagnosis / prognosis

```text
tools/launch.py main()
├── tools/_common.resolve_cli_config -> source/utils/config.load_config
├── source/data/paths.ProjectPaths.resolve
├── source/data/preflight.require_preflight -> run_preflight
├── source/distributed/launcher.build_launch_spec   (torchrun khi >1 GPU)
└── launch() => tools/tasks/train_task.py main()
    ├── source/distributed/setup.initialize_distributed
    ├── source/engine/experiment.OutputManager      (run dir, config/lineage/environment)
    ├── tools/_common.build_dataset -> source/data/dataset.CTPADataset   (train, validation)
    ├── source/engine/factory.build_task_model
    │   ├── components/encoders/image/registry.build_image_encoder
    │   │   ├── "ct_fm" / "ct_fm_features" -> model/3D_model/ctfm.py -> image/ct_fm.py
    │   │   └── tên baseline zoo           -> model/registry.baseline_builders
    │   │        -> model/2D_model/*.py (-> builder.build_slice_mil_encoder) | model/3D_model/*.py
    │   ├── components/peft/freeze.apply_peft (-> lora.inject_lora)
    │   └── model/classifier.BaselineClassifier   (task.architecture=baseline_classifier, khác -> lỗi)
    ├── _fit_feature_standardizer                    (chỉ trên train)
    ├── source/distributed/setup.wrap_ddp
    ├── source/engine/trainer.Trainer.fit  (task_steps.task_loss_step -> tasks/*/losses;
    │                                       checkpoint.save_checkpoint_atomic: best.ckpt, last.ckpt)
    ├── source/engine/task_artifacts.write_training_artifacts / write_backbone_previews
    └── source/profiling/model_profile.profile_model
```

### 2.2 Evaluate

```text
tools/launch.py --evaluate => tools/tasks/evaluate.py main()
├── resolve_cli_config -> require_preflight -> OutputManager
├── build_task_model -> load_checkpoint(strict=True)       (best.ckpt của run)
├── tools/_common.build_dataset (validation, test) -> model
├── source/distributed/gather.gather_prediction_rows
├── metrics/result_table.select_threshold (validation) | _locked_external_threshold
├── metrics/result_table.metric_bundle -> classification | prognosis metrics + bootstrap
├── write_result_csv, write_predictions_csv
├── (--reference-predictions) metrics/paired.paired_patient_bootstrap
├── write_backbone_previews (Grad-CAM, visualize/{correct,incorrect}/)
└── metrics/reporting.stard_ai_checklist | tripod_ai_checklist
```

### 2.3 Các stage sinh dữ liệu

```text
tools/data/build_dataset.py -> source/data/build/pipeline.build_dataset
    sources.InspectSource -> adjudication + sampling -> leakage.load_excluded_patients
    -> filters.apply_eligibility -> integrity.check_volumes -> leakage.require_no_leakage
    -> volumes.preprocess_study -> ehr.build_ehr_profiles -> spesi.build_spesi_artifacts
    -> manifest_writer.build_manifests -> quality.build_data_quality (data_quality.md)

tools/create_masks/generate_masks.py -> source/segmentation/pipeline.generate_pseudo_anatomy
    resume.guard_resume_settings; _process_study (thread / GPU):
    totalsegmentator.TotalSegmentatorRunner (total, heartchambers_highres, lung_vessels)
    + lungmask.LungMaskRunner (tùy chọn) -> imaging/nifti.mask_qc, binary_dice -> imaging/preview

tools/build_rois/build_rois.py -> source/roi/builder.build_roi_dataset
    _process_study (thread / CPU): roi/masks (union, dilate, subtract, body mask)
    -> random_controls.matched_random_control (ROI8) -> imaging/preview.write_overlay_previews

tools/launch.py (silver) => tools/silver_labels/generate_silver_labels.py
    chia report theo rank -> silver/providers.TransformersProvider -> medgemma.MedGemmaExtractor
    -> silver/generator.SilverGenerator (rules.apply_rule, confidence.route_confidence,
       audit.AuditRecord, schema.validate_target_value) -> distributed/gather -> silver/qc
```

### 2.4 Các luồng phụ

```text
scripts/tool/run_baseline_grid.sh => tools/baselines/run_many.py
    => run_case.py (mỗi case một tiến trình / GPU slot)
        case_manifest -> source/data/experiment_splits.ExperimentSplits.materialize
                      -> tools/baselines/fraction_subsets.py (exp02: xuất subset + check.json)
        => tools/preflight.py | run_status.py | launch.py | launch.py --evaluate
    => tools/baselines/summarize.py

tools/tasks/zeroshot_penet.py -> components/encoders/image/penet_zeroshot -> imaging/penet_preview
tools/tasks/zeroshot_radar.py => zeroshot_radar_worker.py   (trong conda env RADAR)

scripts/diagnosis/baselines/prepare_ctfm_cache.sh => tools/data/build_ctfm_cache.py
    (một lần mỗi profile, trước mọi case ctfm_frozen_3d)

tools/baselines/smoke_pipeline.py => smoke_data.py (dữ liệu tổng hợp) => run_case.py --smoke
```

## 3. Abstraction trung tâm

| Abstraction | File | Vai trò |
|---|---|---|
| `ProjectPaths` | `source/data/paths.py` | Một chỗ quyết định mọi root (`raw_inspect_root`, `derived_root`, `output_root`, `cache_root`, ...) từ config + `PE_*`; `require_within_output` chặn ghi ra ngoài output root |
| `OutputManager` | `source/engine/experiment.py` | Tạo/kiểm tra run dir mọi stage, ghi `resolved_config`, lineage, result; chặn ghi đè nếu không có `--overwrite`/`--resume` |
| `load_config` | `source/utils/config.py` | YAML kế thừa (`_base_`, `file.yaml#preset`), `--set`, biến môi trường, `validate_config` (điền backbone, chọn init, stamp variant) |
| `read_rows` | `source/data/manifests.py` | Đọc manifest CSV / JSONL / Parquet; gần như mọi stage đọc dữ liệu qua đây |
| `require_preflight` | `source/data/preflight.py` | Cổng kiểm tra trước khi chạy (PROJECT, DATA, MODEL, SUPERVISION, LINEAGE, COMPUTE, OUTPUT); mọi CLI gọi trước khi ghi gì |
| `BaseImageEncoder` | `source/components/encoders/image/base.py` | Hợp đồng của mọi image encoder: `forward_features` trả `ImageFeatures` (feature map + global embedding) |
| `BaselineEncoder` | `source/model/base.py` | Bọc "core" của baseline zoo, thêm `IntensityAdapter` và báo cáo nạp pretrained weight |
| `build_image_encoder` | `source/components/encoders/image/registry.py` | Ánh xạ `model.backbone` -> builder (CT-FM, toàn bộ zoo) |
| `build_task_model` | `source/engine/factory.py` | Dựng `BaselineClassifier` + PEFT từ config (kiến trúc trainable duy nhất); dùng bởi train, evaluate, Grad-CAM |
| `Trainer` | `source/engine/trainer.py` | Vòng lặp chung: AMP, grad accumulation, DDP reduce, scheduler, early stopping, `history.csv`, checkpoint |
| `CTPADataset` | `source/data/dataset.py` | Dataset trên manifest: `volume`, `labels`, `label_valid` (+ trường tùy chọn) |
| `metric_bundle` | `source/metrics/result_table.py` | Metric + CI bootstrap cấp patient cho một split; dùng chung evaluate và zero-shot |

## 4. File quan trọng theo package

### 4.1 Gốc và `tools/`

Cách chạy: [05_running_outputs.md](05_running_outputs.md).

| File | Vai trò |
|---|---|
| `run.py` | Menu theo tên experiment: đọc registry, in thông tin/kế hoạch, kiểm tra cờ scope, chuyển sang CLI đúng stage |
| `tools/_common.py` | Parser chuẩn (`--config/--set/--gpus/--resume/--overwrite`), `resolve_cli_config`, `build_dataset`, `build_training_lineage`, `import_symbol` |
| `tools/launch.py` | Chạy một experiment trên CPU/1 GPU/DDP: kiểm tra stage và scope, preflight, rồi gọi entrypoint (train, evaluate, silver) |
| `tools/preflight.py` | In báo cáo preflight JSON cho một config, exit 2 nếu có FAIL |
| `tools/run_status.py` | Trạng thái một task run (`absent`, `incomplete`, `trained`, `evaluated`) và settings có khác config không |
| `tools/data/build_dataset.py` | Build dataset profile ([01_data_pipeline.md](01_data_pipeline.md)) |
| `tools/data/build_ctfm_cache.py` | Chạy CT-FM frozen theo patch, lưu feature grid + pooled, viết `manifests/ct_fm/*.csv` |
| `tools/data/build_split_manifests.py` | Manifest theo tỷ lệ train (exp02) cho baseline |
| `tools/create_masks/generate_masks.py`, `tools/build_rois/build_rois.py`, `tools/silver_labels/generate_silver_labels.py` | CLI segmentation / ROI / silver: preflight, resume an toàn, ghi QC |

### 4.2 `tools/tasks/` và `tools/baselines/`

Train/evaluate: [03_training_evaluation.md](03_training_evaluation.md); các arm: [04_experiments.md](04_experiments.md).

| File | Vai trò |
|---|---|
| `tools/tasks/train_task.py` | Train diagnosis/prognosis: dataset, model, fit standardizer trên train, `Trainer`, epoch bundle, profile |
| `tools/tasks/evaluate.py` | Suy luận validation/test, chọn threshold, metric + bootstrap CI, paired bootstrap (`--reference-predictions`), Grad-CAM preview, checklist STARD/TRIPOD |
| `tools/tasks/zeroshot_penet.py`, `zeroshot_radar.py` | Zero-shot PENet / RADAR (RADAR gọi `zeroshot_radar_worker.py` trong env riêng) |
| `tools/baselines/experiments.py` | Lưới model × head × fraction × variant của exp01–04 (đọc `scripts/diagnosis/baselines/<exp>/experiment.yaml`) |
| `tools/baselines/run_case.py`, `run_many.py` | Một case (manifest fraction → preflight/train/evaluate) / cả lưới song song theo slot GPU |
| `tools/baselines/fraction_subsets.py` | Xuất subset train exp02 theo seed ra `splits/data_fraction/seed_<s>/` + `check.json`; `--dir` kiểm lại một thư mục |
| `tools/baselines/summarize.py`, `prepare_weights.py`, `smoke.py` | Bảng + biểu đồ kết quả / tải trước pretrained weight / một bước train thật cho từng arm |
| `tools/baselines/smoke_data.py`, `smoke_pipeline.py` | Dataset tổng hợp nhỏ (không dữ liệu bệnh nhân) / chạy toàn pipeline trên nó (CPU được, `output/_smoke` hoặc `$PE_SMOKE_ROOT`) |

### 4.3 `source/utils/`, `source/data/`

| File | Vai trò |
|---|---|
| `utils/config.py` | `load_config`, `validate_config`, `deep_merge`, `apply_overrides`, `resolve_backbone`, `infer_compute_strategy` |
| `utils/workers.py`, `seed.py`, `environment.py`, `logger.py` | Số worker theo CPU/RAM, seed, git state + phiên bản thư viện, `RunLogger` |
| `data/paths.py` | `ProjectPaths` |
| `data/manifests.py` | `read_rows`, audit manifest/report/silver/temporal holdout |
| `data/dataset.py` | `CTPADataset` (đọc volume cache kèm kiểm sidecar, nhãn) |
| `data/preflight.py` | `run_preflight`, `require_preflight`, `checkpoint_lineage_errors` |
| `data/experiment_splits.py` | Subset train cấp patient (stratify, lồng nhau, theo seed) cho exp02 |
| `data/profiles/__init__.py` | Đọc dataset profile (YAML kế thừa), `require_active_profile`, `assert_shared_preprocessing` |

### 4.4 `source/data/build/` (stage 0)

| File | Vai trò |
|---|---|
| `pipeline.py` | `build_dataset`: điều phối cả lần build, ghi `dataset.json` |
| `sources.py`, `sampling.py` | Join bảng INSPECT thành `StudyRecord`; lấy mẫu patient phân tầng trong từng official split |
| `filters.py`, `integrity.py` | Eligibility + sổ study bị loại; phát hiện CT hỏng/thiếu |
| `adjudication.py`, `leakage.py` | Gộp nhãn cấp patient (mortality event/censored); giữ official split, kiểm tra leakage, governance exclusion |
| `volumes.py` | Tiền xử lý CT (resample, cửa sổ HU, crop/pad, sidecar), cache atomic |
| `ehr.py`, `spesi.py` | Feature EHR trước CTPA (không leakage); tính sPESI từ MEDS/OMOP |
| `manifest_writer.py`, `quality.py` | Viết `manifests/*.csv`; báo cáo `data_quality.md` |

### 4.5 `source/imaging/`, `source/segmentation/`, `source/roi/`

| File | Vai trò |
|---|---|
| `imaging/nifti.py` | Đọc/ghi NIfTI, Dice, `mask_qc` |
| `imaging/grid.py` | Đưa mask lên lưới input model đã tiền xử lý (đọc sidecar) |
| `imaging/preview.py`, `cam_preview.py`, `penet_preview.py` | Overlay ROI / contact sheet segmentation; Grad-CAM HTML/PNG; Grad-CAM riêng cho PENet |
| `segmentation/pipeline.py` | `generate_pseudo_anatomy`: song song theo GPU, cache trạng thái, QC |
| `segmentation/totalsegmentator.py`, `lungmask.py` | Bọc TotalSegmentator / LungMask trong tiến trình con |
| `segmentation/resume.py` | Fingerprint setting ảnh hưởng output; từ chối resume nếu khác (dùng cả cho ROI) |
| `roi/registry.py` | 8 ROI (mã, tên, vai trò KEEP_ONLY/REMOVE_ROI) |
| `roi/builder.py`, `masks.py` | Dựng ROI từ mask segmentation (union/dilate/subtract); phép toán mask |
| `roi/random_controls.py` | ROI8: tịnh tiến cứng ROI tới vị trí ngẫu nhiên tái lập được trong cơ thể |

### 4.6 `source/silver/`, `source/clinical/`

| File | Vai trò |
|---|---|
| `silver/generator.py` | `SilverGenerator`: MedGemma gán nhãn, regex làm dự phòng/cứu, ép nhất quán PE, audit |
| `silver/rules.py` | Luật regex xử lý section, mệnh đề, phủ định, không chắc chắn |
| `silver/providers.py`, `extractor.py`, `medgemma.py` | Provider Hugging Face + prompt/parse JSON; hỏi một target mỗi lần theo hợp đồng JSON; `MedGemmaExtractor` |
| `silver/schema.py`, `confidence.py`, `audit.py`, `qc.py` | Target + kiểu giá trị; ngưỡng confidence; bản ghi audit; QC |
| `clinical/spesi.py` | `compute_spesi`, báo rõ thành phần thiếu (dùng khi build dataset) |

### 4.7 `source/components/`

Chi tiết kiến trúc: [02_models.md](02_models.md).

| File | Vai trò |
|---|---|
| `targets.py` | `TargetSpec`, loss có mask cho đa nhiệm |
| `adapters/standardization.py` | `PooledFeatureStandardizer`: z-score feature pooled, fit trên train |
| `encoders/image/base.py`, `registry.py` | Hợp đồng `BaseImageEncoder`; `build_image_encoder` |
| `encoders/image/ct_fm.py` | CT-FM: đổi hướng RAS↔SPL, thang HU, feature theo patch, `CachedCTFMEncoder` |
| `encoders/image/penet_zeroshot.py` | Nạp PENet đã phát hành + cửa sổ 32 lát cho zero-shot |
| `encoders/image/external.py` | Bọc encoder bên thứ ba sau khi kiểm tra API, nạp state file |
| `peft/freeze.py`, `lora.py` | Chiến lược fine-tune (full, frozen, LoRA); `inject_lora` |

### 4.8 `source/model/`, `source/tasks/`

| File | Vai trò |
|---|---|
| `model/registry.py` | Tên backbone baseline -> (module, builder), nạp lười bằng `importlib` |
| `model/base.py` | `BaselineEncoder`, `IntensityAdapter`, `load_state_with_report` (báo khớp/thiếu weight) |
| `model/classifier.py` | `BaselineClassifier`: encoder -> projection chung -> head |
| `model/inflate.py`, `weights.py` | Inflate weight 2D -> 3D kiểu I3D; vị trí và tải weight baseline |
| `model/2D_model/builder.py`, `mil.py` | Builder chung 2D/2.5D slice-MIL; `SliceMILCore` + `GatedAttentionPool` |
| `model/2D_model/{resnet,convnext,vit,swin}.py` | Backbone slice 2D |
| `model/3D_model/{resnet,densenet,convnext,vit,swin}.py` | Backbone 3D (MONAI / tự viết / Swin-UNETR) |
| `model/3D_model/{nnmamba,mamba_mae,vmamba}.py` | Họ Mamba 3D (nnMamba4cls, Mamba-MAE, VMamba-v2) |
| `model/3D_model/penet.py`, `ctfm.py` | PENet fine-tune từ weight PE; hai arm CT-FM (LoRA, frozen cached) |
| `model/3D_model/_thirdparty.py` | Import module từ `third_party/repos` không sửa chúng |
| `model/head/{factory,mlp,kan}.py` | Chọn head theo `head.type`; MLP; KAN (pykan vendored) |
| `tasks/diagnosis/losses.py` | `diagnosis_loss`: BCE / CE có mask theo target (prognosis dùng `targets.masked_multitask_loss`) |

### 4.9 `source/engine/`, `source/metrics/`, `source/distributed/`, `source/profiling/`

Chi tiết: [03_training_evaluation.md](03_training_evaluation.md).

| File | Vai trò |
|---|---|
| `engine/factory.py` | `build_task_model` |
| `engine/trainer.py`, `schedulers.py` | `Trainer`; warm-up tuyến tính + cosine |
| `engine/task_steps.py` | Loss một bước diagnosis/prognosis |
| `engine/checkpoint.py` | Lưu/nạp checkpoint atomic kèm lineage + SHA-256 (giữ trường hợp đồng `fold`, `fusion_type`, `organ_adapter_configuration`) |
| `engine/experiment.py` | `OutputManager`, `prepare_resumable_run`, `run_output_id`, `compact_result` |
| `engine/task_artifacts.py` | Epoch bundle (đường cong, log), `cleanup_task_run`, `write_backbone_previews` |
| `metrics/classification.py`, `prognosis.py`, `calibration.py` | Metric nhị phân, metric prognosis, calibration slope/intercept |
| `metrics/bootstrap.py`, `paired.py` | Bootstrap theo patient; paired bootstrap so với reference |
| `metrics/result_table.py` | `result.csv`, `predictions.csv`, chọn threshold, `metric_bundle` |
| `metrics/reporting.py` | Checklist STARD-AI / TRIPOD-AI |
| `distributed/setup.py`, `launcher.py`, `gather.py` | Process group + DDP; dựng lệnh python/torchrun; gom prediction mọi rank |
| `profiling/model_profile.py` | Latency, FLOPs, VRAM đỉnh |

### 4.10 `analysis/`

EDA chỉ đọc artifact đã build, chạy qua `scripts/data/eda.sh`: `run_eda.py` (CLI), `loaders.py` (nạp bảng, ghi nhận bảng thiếu), `cohort.py`, `labels.py`, `outcomes.py`, `geometry.py`, `reports.py` (độ phủ luật silver), `figures.py`, `report.py` (Markdown).

## 5. Shell entry points

Mọi script `source scripts/tool/use_gcs_storage.sh`; nhiều script dùng thêm `_flags.sh` (`is_true`/`is_false`).

| Script | Python được gọi |
|---|---|
| `scripts/data/preprocessing.sh` | `run.py <ACTION> data.dataset.<PROFILE>` -> `tools/data/build_dataset.py` |
| `scripts/data/segmentation.sh`, `roi.sh` | `run.py run data.segmentation` / `data.roi` |
| `scripts/data/silver_labels.sh` | `run.py run data.silver.medgemma` -> `tools/launch.py` |
| `scripts/data/eda.sh` | `analysis/run_eda.py` |
| `scripts/diagnosis/zero_shot/{penet,radar}.sh` | `tools/tasks/zeroshot_penet.py` / `zeroshot_radar.py` |
| `scripts/diagnosis/baselines/exp0*/**/<model>.sh`, `run_all.sh`; `scripts/prognosis/baselines/exp0*.sh` (`--label`) | `scripts/tool/run_baseline_grid.sh <exp> <model\|all\|dim:<dim>>` -> `tools/baselines/run_many.py` |
| `scripts/diagnosis/baselines/prepare_ctfm_cache.sh` | `tools/data/build_ctfm_cache.py` (cache CT-FM cho `ctfm_frozen_3d`) |
| `scripts/diagnosis/baselines/{summarize,prepare_weights,smoke}.sh` | `tools/baselines/{summarize,prepare_weights,smoke}.py` |
| `scripts/tool/use_gcs_storage.sh`, `_flags.sh` | (không gọi Python) đặt biến storage; parse boolean |

## Xem thêm

[README.md](README.md) (tổng quan), [05_running_outputs.md](05_running_outputs.md) (chạy + output), [04_experiments.md](04_experiments.md) (danh sách thí nghiệm), `python run.py list`.
