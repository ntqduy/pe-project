# 03. Huấn luyện và đánh giá (cơ chế chung)

> **Lưu ý (2026-10-03):** các experiment anatomy-aware / Soft-MoE / late-logit / global / matrix, toàn bộ `03_prognosis` (modality, EHR ablation), `04_anatomy_analysis` (counterfactual, architecture) và external Turkey test đã được gỡ khỏi `configs/` và `configs/experiments.yaml`; phần nhắc tới chúng dưới đây chỉ còn giá trị lịch sử (config cũ: `git show 41e4c8c:configs/runs/...`). Protocol đang dùng: baseline zoo exp01-exp04 (`scripts/diagnosis/baselines/README.md`), CT-FM frozen, zero-shot PENet/RADAR và pipeline dữ liệu `00_data`.

**Đọc khi:** muốn hiểu một run diagnosis/prognosis chạy thế nào từ lệnh đến `result.csv`: chọn GPU, DDP, vòng epoch, chọn checkpoint, chọn threshold, bootstrap CI, so sánh cặp, Grad-CAM preview, resume/overwrite.

**Code chính:** `tools/launch.py`, `tools/tasks/train_task.py`, `source/engine/{trainer,checkpoint,transfer,task_artifacts,experiment}.py`, `tools/tasks/evaluate.py`, `source/distributed/{setup,gather,launcher}.py`, `source/metrics/{classification,bootstrap,calibration,paired,result_table,reporting}.py`, `tools/run_status.py`, `source/profiling/model_profile.py`

Đây là phần máy móc dùng chung cho mọi task. Nội dung riêng của từng task và danh sách experiment ở [04_experiments.md](04_experiments.md); kiến trúc model ở [02_models.md](02_models.md); thư mục output ở [05_running_outputs.md](05_running_outputs.md).

---

## 1. Luồng tổng thể

```text
python run.py run <exp> --gpus 0,1           (run.py tra configs/experiments.yaml -> config path)
        │
        ▼
tools/launch.py  ── preflight ── chọn CPU / 1 GPU / torchrun DDP
        │
        ├── (mặc định)   tools/tasks/train_task.py  -> Trainer.fit -> best.ckpt/last.ckpt
        │                                             -> epoch_<E>/{checkpoint/, history.csv, logs.txt,
        │                                                training_curves.png, preview/, result.json}
        └── (--evaluate) tools/tasks/evaluate.py    -> threshold trên validation -> test
                                                      -> epoch_<E>/{result.csv, predictions.csv,
                                                         result.json (+evaluation), preview/,
                                                         reporting_checklist.json}
```

Train và evaluate là **hai process riêng**: `python run.py run` chỉ train; evaluate gọi riêng (`tools/launch.py --evaluate` hoặc `tools/tasks/evaluate.py`). Các wrapper (`scripts/tool/run_ctfm_frozen.sh`, `scripts/tool/run_baseline_grid.sh` → `tools/baselines/run_case.py`) với `ACTION=all` gọi lần lượt cả hai. Metric task chỉ được tính ở `evaluate.py`; training chỉ ghi loss và AUROC theo epoch để vẽ đường cong.

---

## 2. `tools/launch.py`: chọn thiết bị và chế độ chạy

- **Resolve config:** `resolve_cli_config` (`tools/_common.py`) = `load_config` + `--set` + `--gpus` → `compute.devices`, `compute.strategy` (`[]`→`cpu`, 1 GPU→`single`, ≥2→`ddp`).
- **Entrypoint** (`ENTRYPOINTS`): `diagnosis`/`prognosis` → `train_task.py`; `silver` → `generate_silver_labels.py`; `counterfactual` → `counterfactual.py`; `--evaluate` → `evaluate.py`.
- **Chặn sai cách dùng:** `experiment.family == remove_roi` bị từ chối; cờ chọn ca (`--patient-id`, `--max-cases`, `--max-reports`, `--allow-full`) chỉ hợp lệ cho silver/counterfactual/evaluate.
- **Preflight** (`require_preflight`, `source/data/preflight.py`) chạy ở process cha, fail sớm.
- **Map GPU:** đặt `CUDA_VISIBLE_DEVICES=<GPU vật lý>`, process con thấy `compute.devices=[0..n-1]`.
- **Spawn** (`build_launch_spec`): ≤1 GPU → `python <entrypoint>`; ≥2 GPU → `torchrun --standalone --nproc-per-node=<n>` (fallback `python -m torch.distributed.run`).
- **Evaluate** mà không `--overwrite` → `config["resume"]=True` để đọc lại thư mục run đã train.

```bash
python tools/launch.py --config configs/runs/02_diagnosis/global/global_single.yaml --gpus 0,1 --dry-run
python tools/launch.py --config configs/runs/02_diagnosis/global/global_single.yaml --gpus 0,1
python tools/launch.py --config configs/runs/02_diagnosis/global/global_single.yaml --gpus 0,1 --evaluate --allow-full
```

Trong process con, `initialize_distributed` (`source/distributed/setup.py`) đọc `WORLD_SIZE`/`RANK`/`LOCAL_RANK`, `init_process_group` (`nccl` nếu có CUDA, `gloo` nếu không; `compute.distributed_backend`) và trả `DistributedContext`. `rank_zero_call` chạy side effect (preflight, tạo thư mục) ở rank 0 rồi broadcast kết quả hoặc lỗi, nên mọi rank cùng fail thay vì treo.

---

## 3. `train_task.py`: các bước

1. Từ chối `--resume` (task training không resume), và config có `external_evaluation.test_only`/`prohibit_training`.
2. Init DDP, preflight (rank 0).
3. Thư mục run: `run_output_id(config)` = `<experiment.id>/epoch_<training.epochs>`; `OutputManager.prepare` kiểm collision, ghi `resolved_config.yaml`.
4. `seed_everything(seed)` **giống nhau ở mọi rank** cho tới khi bọc DDP (model khởi tạo và standardizer fit ra cùng giá trị); `seed` mặc định 42.
5. Dataset `CTPADataset` cho train/validation; prognosis bỏ dòng không có nhãn quan sát nào. `data.preload_inputs` nạp feature CT-FM pooled vào RAM.
6. DataLoader: `compute.num_workers` (`auto` bị chặn theo RAM trống và chia cho `world_size`); DDP dùng `DistributedSampler(train, shuffle=True, seed)`.
7. `build_task_model(config)` → `(model, peft_report)`; log số tham số trainable.
8. Transfer nếu có `lineage.source_checkpoint` (mục 5).
9. Fit trên train: `_fit_clinical_preprocessor` (impute/z-score EHR) và `_fit_feature_standardizer` (z-score feature đã pool), chỉ từ split train.
10. `wrap_ddp` (`find_unused_parameters=True` chỉ khi encoder khai báo `ddp_find_unused_parameters`, ví dụ PENet); sau đó `seed_everything(seed + rank)` để augmentation/dropout khác nhau giữa rank.
11. `AdamW(lr, weight_decay)` trên tham số `requires_grad`, scheduler từ `build_scheduler`.
12. `Trainer(...).fit(...)`.
13. Rank 0: `best_epoch(history)` (không có epoch hữu hạn nào → lỗi), load lại `best.ckpt` strict, `write_training_artifacts`, `write_backbone_previews`, `profile_model`, ghi `result.json` (`status=completed`), `cleanup_task_run`.

**Feature standardizer** (`_fit_feature_standardizer`): baseline classifier tự fit khi `head.standardize_inputs: true`; model có `organ_adapters.input_standardizer` (bật bằng `organ_adapter.standardize_inputs: true`) fit một lượt qua train: mỗi rank xử lý một shard rời, `all_reduce` tổng/tổng bình phương/số dòng, nên mọi rank có cùng `μ`, `σ` trước khi bọc DDP. `μ`, `σ` là buffer trong `state_dict` nên nằm trong checkpoint. Nhánh có < 2 dòng train không chuẩn hoá được (ghi `identity_branches` + `WARNING`). Bật mà model không hỗ trợ → `ValueError`.

Seed khác: `apply_counterfactual` dùng `seed` (+ patient/study id); `evaluate.py` dùng `seed + rank`, bootstrap dùng `seed`.

---

## 4. `Trainer` (`source/engine/trainer.py`)

`Trainer` không biết task: forward/loss nằm trong `loss_step` do `task_loss_step(config)` trả về (`loss` hoặc `(loss, metrics)`).

```text
for epoch in 1..E:
    train_epoch -> train_loss ;  validation_loss (no_grad) -> val_loss
    primary = -val_loss
    epoch_metrics_fn -> train/val AUROC, AUPRC   (nếu training.record_epoch_auc)
    scheduler.step(primary) ;  improved = primary > best
    rank 0: ghi logs/history.csv; nếu improved -> lưu best.ckpt
    barrier ; early stopping? -> break
rank 0: lưu last.ckpt
```

- **AMP:** `compute.precision` = `bf16` (mặc định) / `fp16` / `fp32` / `auto` (baseline zoo: bf16 nếu GPU hỗ trợ, không thì fp16, CPU thì fp32; `resolve_precision` trong `source/engine/trainer.py`, `run.log` ghi giá trị thực dùng); autocast chỉ bật trên CUDA với bf16/fp16, `GradScaler` chỉ cho fp16.
- **Gradient accumulation** (`training.gradient_accumulation = K`): loss chia cho **kích thước thật của nhóm** (`min(K, len(loader) - group_start)`), nên nhóm cuối epoch ngắn hơn K vẫn là trung bình đúng; `optimizer.step()` mỗi K batch hoặc ở batch cuối. Batch hiệu dụng = `batch_size × K × số GPU`. DDP không dùng `no_sync`, nên gradient all-reduce ở mọi micro-step (đúng, chỉ tốn giao tiếp).
- **Reduce qua rank:** loss trung bình qua `all_reduce`. Metric phụ (`main.<target>.loss`, `auxiliary.*`, `total_loss`) `all_gather_object` tên của mọi rank, lấy hợp rồi một `all_reduce` duy nhất, vì một rank có thể thiếu target không có nhãn trong batch. AUROC theo epoch: `gather_prediction_rows` về rank 0 tính `roc_auc_score`/`average_precision_score`, rồi broadcast; target một lớp → NaN. Pass này đọc lại cả train + validation mỗi epoch; tắt bằng `training.record_epoch_auc: false`.
- **Chọn checkpoint:** mặc định theo **`-val_loss`** (`selection_metric: negative_validation_loss` trong `result.json`). `training.selection_metric: val_auroc` (mọi baseline zoo, `baselines.yaml`) chọn theo **val AUROC** của target chính (`validation_auroc` trong `result.json`); khi đó pass AUROC validation chạy mỗi epoch kể cả khi `record_epoch_auc: false` (chỉ bỏ pass train). `best.ckpt` = epoch đầu tiên có metric cao nhất; NaN không bao giờ "cải thiện".
- **Seed:** `seed_everything` (python / numpy / torch / CUDA, cuDNN deterministic) và DataLoader có `generator` + `worker_init_fn` (`source/utils/seed.py:loader_seeding`), nên thứ tự shuffle và RNG của worker lặp lại theo seed.
- **Early stopping:** `training.early_stopping_patience: P` dừng khi P epoch liên tiếp không cải thiện (`null` = chạy hết). Quyết định tính từ giá trị đã reduce nên mọi rank dừng cùng lúc. `result.json` → `evaluation.training` ghi `epochs_configured`, `epochs_run`, `early_stopping_patience`, `stopped_early`. Patience phải giống nhau cho mọi arm của một so sánh.
- **Scheduler:** `training.scheduler` = `none` (mặc định) hoặc `cosine` (warm-up tuyến tính `warmup_epochs`, rồi cosine xuống `min_lr_factor × lr`), gọi mỗi epoch.
- **`history.csv`** (ghi atomic mỗi epoch): `epoch`, `train_loss`, `val_loss`, `primary_val_metric` (= `-val_loss`), `lr`, `epoch_time_sec`, `peak_vram_gb`, `train_<metric>`/`val_<metric>` từ `loss_step` (ví dụ `val_total_loss`), và `train_/val_<target>_auroc`, `..._auprc`.

---

## 5. Checkpoint, transfer, `encoder.init_source`

**Định dạng** (`source/engine/checkpoint.py`): `save_checkpoint_atomic` ghi dict `schema_version (=2), lineage, model_state (bỏ "module."), optimizer_state, scheduler_state, extra` vào file tạm, `fsync`, `os.replace`, rồi **load lại để kiểm lineage**; ghi thêm `<ckpt>.metadata.json` (`checkpoint_id` = SHA-256 của file...). `lineage` phải đủ `REQUIRED_LINEAGE` (experiment_id, stage, backbone, initialization, source_checkpoint, source_checkpoint_hash, dataset, split, task, fusion_type, random_seed, code_commit, epoch, validation_metric...); có `source_checkpoint` thì hash phải là SHA-256 hex. `load_checkpoint(path, model, strict=True)` kiểm lineage và shape, thiếu/thừa key là lỗi.

**Transfer một phần** (`source/engine/transfer.py`): `transfer_modules(model, ckpt, modules)` chỉ lấy key có tiền tố của module chọn (mặc định `image_encoder.`), vẫn strict trong phạm vi đó; head và fusion giữ khởi tạo mới.

**Nguồn khởi tạo encoder** (`resolve_encoder_initialization`, preset ở `configs/components/encoders.yaml`):

| `encoder.init_source` | Load gì | `lineage.initialization` |
|---|---|---|
| `pretrained` (alias `public`, `published`, `original`) | trọng số public qua contract backbone; cấm kèm checkpoint | `public` |
| `diagnosis` (alias `c_diagnosis`) | transfer `lineage.transfer_modules` từ `encoder.sources.diagnosis` (= `DX_anatomy_concat/epoch_50/checkpoint/best.ckpt`) | `C_diagnosis` |
| `custom` | bắt buộc `encoder.checkpoint`, nên kèm `encoder.source_experiment` | `custom` |

Checkpoint mặc định của một nguồn chỉ dùng cho `experiment.baseline_backbone` tương ứng, backbone khác → `ConfigError`. `stamp_experiment_variant` thêm hậu tố vào run ID khi lệch baseline: `__ds_<profile>` (profile ≠ `full_inspect`), `__bb_<backbone>`, `__enc_<init_source>`; `experiment.variant_stamp: false` tắt việc này. Vì vậy evaluate phải lặp lại đúng các `--set` đã dùng khi train.

```bash
python tools/tasks/train_task.py --config configs/runs/02_diagnosis/global/global_single.yaml \
  --set encoder.init_source=custom --set encoder.checkpoint=<path/best.ckpt> \
  --set encoder.source_experiment=<RUN_ID> --gpus 0
```

---

## 6. Bundle `epoch_<E>` và dọn dẹp

```text
<outputs>/<family>/<experiment.id>/epoch_<training.epochs>/
├── resolved_config.yaml
├── result.json                 training ghi; evaluate bổ sung evaluation, evaluation_checkpoint
├── checkpoint/best.ckpt  last.ckpt     best = epoch tốt nhất theo -val_loss; last = epoch cuối thực chạy
├── history.csv  logs.txt  training_curves.png
├── preview/                    Grad-CAM (training ghi bản chưa threshold; evaluate ghi đè)
├── result.csv  predictions.csv  reporting_checklist.json     evaluate ghi
├── bootstrap_metrics.parquet   chỉ khi --reference-predictions
└── smoke/<sha12>/              evaluate với --patient-id/--max-cases
```

- `E` là **ngân sách** `training.epochs`, không phải số epoch thực chạy; mỗi ngân sách là một run độc lập. Run test-only, `remove_roi`, `external_zero_shot` không có `epoch_<E>`.
- `cleanup_task_run` xoá bản sao ở gốc (`best.ckpt`, `last.ckpt`, `config.yaml`, `metrics.json`...), thư mục `logs/`, `checkpoints/`, `figures/`, `qc/`, `plots/` và file legacy sau khi bundle đã có `checkpoint/`; không xoá `bootstrap_metrics.parquet`/`reporting_checklist.json`.
- Train copy `logs/run.log` → `logs.txt`; evaluate **append** vào `logs.txt`.
- Sau train, `profile_model` đo latency/GFLOPs/VRAM trên một batch validation ở `eval()` (1 warm-up + `profiling.iterations`, mặc định 5), ghi vào `result.json` (`model.latency_ms_per_volume`, `gflops_per_volume`, `compute.peak_vram_gb`, `compute.training_time_min`).

---

## 7. `evaluate.py`

1. **External test-only** (`external_evaluation.test_only`): bắt buộc `--checkpoint`, `threshold_source: internal_validation_artifact`, `evaluation_split: test`.
2. Dùng cùng `run_output_id` như train; thư mục không tồn tại mà không có `--checkpoint` → lỗi "train with the same training.epochs first".
3. Checkpoint: `--checkpoint` hoặc `best.ckpt` của run (gốc → `checkpoint/` → `epoch_*/checkpoint/` cao nhất); load **strict**, rồi `wrap_ddp`.
4. Target: diagnosis chấm primary trước, cộng các head native binary khác có cột nhãn (silver/multiclass/head organ phụ bị bỏ); prognosis chấm các `task.targets` có trong `data.label_columns`; test-only chỉ primary.
5. Threshold (xem dưới), dự đoán test, rank 0 tính `metric_bundle` (train/validation không CI, test có CI).
6. Ghi `result.csv`, `predictions.csv`, `result.json` (giữ `evaluation.training`, thêm `evaluation_scope`, `evaluation_checkpoint` gồm path, sha256, lineage), preview, `reporting_checklist.json` (`stard_ai_checklist` hoặc `tripod_ai_checklist`: chỉ là khung trỏ tới bằng chứng, không phải tuyên bố tuân thủ).

**Gom dự đoán dưới DDP** (`source/distributed/gather.py`): `DistributedSampler` đệm shard cuối bằng dòng lặp, nên `gather_prediction_rows` gom về rank 0, khử trùng lặp theo `(patient_id, study_id)` (bản trùng phải giống nhau, nếu không lỗi) và so với tập dòng có nhãn hợp lệ (thiếu/thừa → lỗi). Nhãn NaN (ví dụ censored) bị bỏ, không đổi thành 0.

### Threshold chỉ chọn trên validation

`select_threshold` chọn threshold cho từng target trên **validation**: `evaluation.threshold_method` = `youden` (mặc định, tối đa `TPR − FPR`) hoặc `f1`. Validation một lớp → 0.5 với `threshold_source = fallback_0.5_validation_single_class`. Threshold được áp **nguyên vẹn** cho train, validation, test. Checkpoint chọn theo `-val_loss` và threshold chọn theo Youden là hai lựa chọn độc lập, đều chỉ dùng validation.

### Threshold khoá cho external test

`_locked_external_threshold` đọc `external_evaluation.threshold_artifact` (một `result.json`, một bundle `epoch_<E>`, hoặc thư mục run; không đặt thì dùng `result.json` cạnh checkpoint) và chỉ chấp nhận artifact có `evaluation_checkpoint.sha256` **bằng SHA-256 của `--checkpoint`** (không khớp → lỗi; nhiều artifact cùng khớp → "ambiguous"), cùng `evaluation.threshold` và `evaluation.threshold_source == "validation"`. Nhãn external nhờ đó không bao giờ chọn threshold.

### Smoke vs full scope

| Scope | Kích hoạt | Ghi ở đâu |
|---|---|---|
| full test | không có `--patient-id`/`--max-cases` (`--allow-full` được chấp nhận) | `<bundle>/result.csv`, `predictions.csv`, `result.json`; `evaluation_scope.mode = full_test` |
| smoke | `--patient-id ...` hoặc `--max-cases N` (chỉ giới hạn test) | `<bundle>/smoke/<12 ký tự sha256 của scope>/`; tồn tại rồi thì cần `--overwrite` |

Smoke vẫn ghi preview vào `<bundle>/preview/`, đè preview của lần full-test.

### Bootstrap CI theo bệnh nhân (`source/metrics/bootstrap.py`)

`patient_bootstrap` rút có hoàn lại **bệnh nhân** (không phải study) `evaluation.bootstrap_samples` lần (mặc định 2000), tính lại mọi metric, lấy phân vị CI 95% (RNG `default_rng(seed)`). Cần ≥ 2 bệnh nhân; metric có < `max(20, n_bootstrap // 2)` replicate hữu hạn giữ point estimate, CI rỗng, kèm `ci_note`. CI chỉ tính trên **test**; `result.csv` hiện CI của AUROC/AUPRC, `result.json` giữ CI mọi metric. Một seed chỉ đo dao động lấy mẫu test, không đo dao động training.

### So sánh cặp (`--reference-predictions`, `source/metrics/paired.py`)

Dùng cho counterfactual/ablation hoặc so hai model trên cùng bệnh nhân (chỉ primary target):

1. Đọc `predictions.csv` test của reference.
2. Threshold của reference lấy từ `result.json` cạnh file đó, không có thì dùng cột `y_pred` sẵn có. **Mỗi model dùng threshold validation của chính nó**, không bao giờ áp threshold của model này cho model kia.
3. `align_predictions_by_patient`: hai tập `(patient_id, study_id)` phải giống hệt, ≥ 2 bệnh nhân, `y_true` khớp.
4. `paired_patient_bootstrap`: mỗi replicate rút **cùng** tập bệnh nhân cho hai arm, `delta = comparison − reference`, CI theo phân vị, cùng luật "đủ replicate".
5. Ghi `bootstrap_metrics.parquet` và `result.json` → `evaluation.paired_vs_reference`.

`--restrict-to <csv/parquet>` (cột `patient_id`, tuỳ chọn `study_id`) giới hạn mọi split vào một danh sách ca chung để mọi arm được chấm trên cùng ca (ví dụ `cache/<profile>/clinical/spesi_evaluable.csv`).

### Calibration (chỉ prognosis, `source/metrics/calibration.py`)

`calibration_slope_intercept`: hồi quy logistic gần như không phạt (`C=1e6`) của nhãn trên `logit(p)`. Trả NaN (lý do ở `note`) khi chỉ một lớp, lớp nhỏ hơn có < `evaluation.calibration_min_events` ca (mặc định 10), dự đoán tách hoàn toàn hai lớp, hoặc fit không hội tụ. Calibration curve (10 bin `quantile`) nằm ở `result.json` → `targets.<t>.calibration_curve`.

---

## 8. Metric

`binary_classification_metrics` (`source/metrics/classification.py`): discrimination dùng `y_prob`; metric ngưỡng dùng `y_prob ≥ threshold`.

| Metric | Định nghĩa / khi không xác định |
|---|---|
| `auroc`, `auprc` | sklearn; chỉ một lớp → NaN |
| `sensitivity`, `specificity` | TP/(TP+FN), TN/(TN+FP); mẫu số 0 → NaN |
| `ppv`, `npv` | TP/(TP+FP), TN/(TN+FN); không có dự đoán dương → NaN |
| `f1`, `accuracy`, `balanced_accuracy` | sklearn (`f1` dùng `zero_division=0`; balanced một lớp → NaN) |
| `brier` | `brier_score_loss` |
| `calibration_intercept`, `calibration_slope` | chỉ prognosis |

Xác suất phải hữu hạn và trong [0,1]. `result.csv` (`result_table.py`) có một dòng cho mỗi (target, split), ô không tính được để trống và cột `note` giải thích (một lớp, ít ca `< evaluation.min_class_count_warning` mặc định 5 → "unstable estimate", threshold nằm ngoài khoảng `y_prob`...). `predictions.csv`: `split, target, patient_id, study_id, y_true, y_prob, y_pred`. Train metrics dùng cùng threshold validation, không CI; chỉ để xem overfit, không để báo cáo.

---

## 9. Grad-CAM preview (tóm tắt)

`write_backbone_previews` (`source/engine/task_artifacts.py`) dựng file bằng `source/imaging/cam_preview.py`. Chỉ dùng validation, định tính, chạy ở rank 0 trên module đã unwrap.

- **Khi nào:** cuối training (chưa có threshold, lớp dự đoán N/A) và cuối evaluate (ghi đè, có threshold).
- **Chọn ca:** mặc định 5 study validation đầu; config `preview: {correct: N, wrong: M}` (baseline zoo dùng 3/3) chọn N ca đúng và M ca sai tự tin nhất theo `|p − threshold|`; `preview.nifti: true` ghi thêm `_ct.nii.gz` + `_gradcam.nii.gz`.
- **Cách tính:** `backward()` từ logit của primary target, hook feature map 5D sâu nhất của backbone. Mặc định `hirescam`, tuỳ chọn `gradcam`. Không đổi weights.
- **File:** `NN_<patient>_<study>_<TP|TN|FP|FN>.html` (viewer offline có thanh trượt z) và `.png`; model 2D/2.5D slice-MIL thêm `_mil_attention.png`.

```bash
python tools/tasks/gradcam_preview.py --run-dir <outputs>/diagnosis/<RUN_ID>/epoch_<E>
# --method gradcam | --maximum-patients 10 | --output <dir> | --checkpoint <ckpt> | --device cpu
```

Chi tiết cho CT-FM cached và PENet/RADAR zero-shot: [04_experiments.md](04_experiments.md).

---

## 10. Resume, overwrite, trạng thái run

| Cờ | Train | Evaluate |
|---|---|---|
| không cờ | `epoch_<E>` đã tồn tại → `OutputCollisionError` | đọc lại run đã train |
| `--resume` | **từ chối** | — |
| `--overwrite` | `rmtree` đúng `epoch_<E>` rồi train lại | ghi đè smoke dir đã có |

`OutputManager.prepare` còn từ chối `--resume` + `--overwrite` cùng lúc, resume run `completed`, và `config_hash` khác run có sẵn. `seed` không được stamp vào run ID, nên nhiều seed cùng config va `OutputCollisionError`.

`tools/run_status.py` (cùng `--config/--set` như `launch.py`) in trạng thái `epoch_<E>`: `absent`; `incomplete` (`result.json` chưa `completed` hoặc thiếu `best.ckpt`); `trained`; `evaluated` (đã có full-test `result.csv` + `predictions.csv`); `different` (`resolved_config.yaml` khác config yêu cầu, bỏ qua `paths`, `config_hash`, `resume`, `overwrite`, `compute.devices`, `compute.num_workers`). Wrapper dùng nó để bỏ qua bước đã xong; `incomplete`/`different` báo lỗi trừ khi `OVERWRITE=1`.

```bash
python tools/run_status.py --config configs/runs/01_foundation/ct_fm_frozen/diagnosis.yaml --format tsv
```

---

## 11. Lệnh thường dùng

```bash
python run.py preflight diag.global.single --gpus 0
python run.py dry diag.global.single --gpus 0,1
python run.py run diag.global.single --gpus 0,1                       # train (DDP 2 GPU)
python tools/launch.py --config configs/runs/02_diagnosis/global/global_single.yaml --gpus 0,1 --evaluate --allow-full
python tools/tasks/evaluate.py --config configs/runs/02_diagnosis/global/global_single.yaml --max-cases 20 --gpus 0   # smoke
python tools/tasks/evaluate.py --config <config> --allow-full \
  --reference-predictions <outputs>/diagnosis/<REF_ID>/epoch_<E>/predictions.csv                                       # so cặp
python run.py run diag.global.single --gpus 0 --set data.profile=smoke_30 --set training.epochs=1                      # rehearse
```
