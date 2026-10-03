# 05. Chạy pipeline và output

> **Lưu ý (2026-10-03):** các experiment anatomy-aware / Soft-MoE / late-logit / global / matrix, toàn bộ `03_prognosis` (modality, EHR ablation), `04_anatomy_analysis` (counterfactual, architecture) và external Turkey test đã được gỡ khỏi `configs/` và `configs/experiments.yaml`; phần nhắc tới chúng dưới đây chỉ còn giá trị lịch sử (config cũ: `git show 41e4c8c:configs/runs/...`). Protocol đang dùng: baseline zoo exp01-exp04 (`scripts/diagnosis/baselines/README.md`), CT-FM frozen, zero-shot PENet/RADAR và pipeline dữ liệu `00_data`.

**Đọc khi:** ngồi vào máy (VM) để chạy thật, hoặc mở một thư mục output và cần biết mỗi file là gì.

**Code chính:** `run.py`, `tools/launch.py`, `scripts/tool/{use_gcs_storage,_flags}.sh`, `configs/paths.yaml`, `source/engine/{experiment,task_artifacts}.py`, `source/metrics/result_table.py`

Mỗi stage làm gì và vì sao: [01_data_pipeline.md](01_data_pipeline.md), [02_models.md](02_models.md), [03_training_evaluation.md](03_training_evaluation.md), [04_experiments.md](04_experiments.md). Cách `run.py` chọn CLI: [README.md](README.md#3-runpy--registry--config--launch--tasks).

## 1. Môi trường và storage

```bash
python -m pip install -r requirements.txt && python -m pip install -e .
gcsfuse --implicit-dirs --only-dir Stanford_INSPECT_dataset pe-study /mnt/Stanford_INSPECT_dataset
```

`configs/paths.yaml` ghi sẵn các root; `ProjectPaths.resolve()` (`source/data/paths.py`) đọc chúng:

| Key | Giá trị | Ghi đè bằng env |
|---|---|---|
| `raw_inspect` | `/mnt/Stanford_INSPECT_dataset` (READ-ONLY) | `PE_RAW_INSPECT_ROOT` (env thắng) |
| `derived_data` | `/mnt/pe-project/outputs/derived` | `PE_DERIVED_ROOT` (env thắng) |
| `output_root` | `/mnt/pe-project/outputs/pe-project/outputs` | không: config thắng |
| `cloud_root` | `/mnt/pe-project/outputs` | không: config thắng `PE_CLOUD_ROOT` |

`PE_LOCAL_CACHE_ROOT` (tuỳ chọn) đổi cache phía code, mặc định `<repo>/cache`. Nên export `PE_CLOUD_ROOT`: `run.py` cảnh báo khi thiếu, và vài config dùng `${PE_CLOUD_ROOT}` (vd `lineage.source_checkpoint` của `anatomy.remove_*`).

`scripts/tool/use_gcs_storage.sh` đặt `PE_CLOUD_ROOT`, `PE_CLOUD_PROJECT_ROOT`, `PE_RAW_INSPECT_ROOT`, `PE_DERIVED_ROOT`; mọi script trong `scripts/` tự `source` nó. Để có cùng biến khi gọi `python run.py`:

```bash
bash scripts/tool/use_gcs_storage.sh --install     # thêm vào ~/.bashrc (--show: in giá trị + kiểm tra mount)
gcloud storage rsync --recursive /mnt/pe-project/outputs gs://pe-study/pe-storage   # backup
```

## 2. `run.py` và quy tắc scope

Lệnh: `list`, `show`, `plan`, `preflight`, `dry` (in lệnh launch), `run`, ví dụ `python run.py run diag.anatomy.concat --gpus 0,1`. Cờ truyền qua: `--gpus`, `--set KEY=VALUE` (lặp được), `--resume` | `--overwrite`, và một cờ scope.

| Stage | Scope bắt buộc | Cờ hợp lệ |
|---|---|---|
| dataset | có | `--max-cases`, `--allow-full` |
| segmentation, roi, counterfactual | có | `--patient-id`, `--max-cases`, `--allow-full` |
| silver | có | `--patient-id`, `--max-reports`, `--allow-full` |
| zero-shot (`runner`) | có | `--max-cases`, `--allow-full` |
| diagnosis / prognosis / ablation (training) | **không** (scope = dataset profile) | không nhận cờ scope |

`dry` không dùng được cho data stage và zero-shot; dùng `preflight`.

## 3. Biến môi trường của `scripts/`

Wrapper nhận cấu hình qua biến đặt trước lệnh (`PROFILE=smoke_30 GPUS=0 bash scripts/data/segmentation.sh`), rồi dịch thành cờ `run.py` / `tools/`:

- `PROFILE`: `smoke_30` | `test_500_sample` | `full_inspect`; mặc định `full_inspect` (`eda.sh`: `test_500_sample`).
- `GPUS`: `0,1,...`; **không đặt** = GPU 0, **đặt rỗng** (`GPUS=`) = CPU.
- `ACTION` (động từ, tuỳ script); `PATIENT_ID`, `MAX_CASES`, `MAX_REPORTS` (scope; không đặt = toàn bộ, `--allow-full`); `PYTHON` (mặc định `python3`).

**Quy ước boolean `=1`** (`OVERWRITE`, `RESUME`, `SMOKE`, `REBUILD_CACHE`, `VERIFY_CACHE`, `CHECK_SLICE_ORDER`, ...; `is_true`/`is_false` trong `scripts/tool/_flags.sh`): bật = `1/true/yes/on`; tắt = không đặt/rỗng/`0/false/no/off`; giá trị khác dừng script với exit 2. Hai cờ mặc định bật, tắt bằng `=0`: `EPOCH_AUC` (baseline grid) và `LUNGMASK` (segmentation).

## 4. Thứ tự chạy: smoke_30 → test_500_sample → full_inspect

Cả ba profile chạy đúng một code path, chỉ cohort khác nhau.

| Bước | Profile | Mục đích | Cách |
|---|---|---|---|
| 1 | `smoke_30` | Kiểm tra kỹ thuật: path, GPU, format output | `PROFILE=smoke_30`, `EPOCHS=1` / `SMOKE=1`, `MAX_CASES=1` cho generation stage |
| 2 | `test_500_sample` | Diễn tập end-to-end, số đo sơ bộ | `--set data.profile=test_500_sample` |
| 3 | `full_inspect` | Kết quả thật cho paper | mặc định mọi config; `--allow-full` cho generation stage |

Trước mỗi bước lớn: `plan`, `preflight`, `dry <exp> --set training.epochs=1`; kiểm tra preview/QC/log của bước nhỏ rồi mới chạy bước lớn. Không sửa `status` trong registry để vượt entry `blocked`.

Phụ thuộc: dataset → segmentation → ROI → `diag.anatomy.*`, `ablation.arch.*` → `anatomy.remove_*`; dataset → silver → `diag.*.silver.*`; dataset → CT-FM cache → `foundation.ct_fm_frozen.*`; còn lại chỉ cần dataset.

## 5. Script theo stage

**Stage 0: dataset.**

```bash
PROFILE=smoke_30 bash scripts/data/preprocessing.sh            # chạy trước; ACTION=run|preflight|plan|show
PROFILE=full_inspect bash scripts/data/preprocessing.sh        # toàn cohort, nhiều giờ
PROFILE=test_500_sample bash scripts/data/eda.sh               # EDA cohort đã build
```

Tương đương `python run.py run data.dataset.<profile> --allow-full`. Profile đã build đủ cùng scope thì script bỏ qua; `OVERWRITE=1` build lại. Đọc `data_quality.md` trước khi chạy tiếp.

**Stage 1: segmentation, ROI.**

```bash
PROFILE=smoke_30 MAX_CASES=1 GPUS=0 bash scripts/data/segmentation.sh   # một study trước
PROFILE=smoke_30 bash scripts/data/roi.sh                               # sau segmentation
```

`roi.sh` tự tìm run `segmentation/SEG_pseudo_anatomy` (thêm `__ds_<profile>` nếu profile ≠ `full_inspect`); `SEGMENTATION_RUN=` ghi đè, `ROI_WORKERS=N` đặt `roi.workers`; `LUNGMASK=0` tắt cross-check. Cả hai mặc định tiếp tục run dở, từ chối nếu setting đổi (`resume_settings.json`), cần `OVERWRITE=1`.

**Stage 2: silver.**

```bash
PROFILE=smoke_30 MAX_REPORTS=10 GPUS=0 bash scripts/data/silver_labels.sh
```

Config silver có `variant_stamp: false`: **mọi profile ghi chung** `silver_label/medgemma/`; đổi profile trên cùng thư mục thì `config_hash` khác và báo collision, phải `OVERWRITE=1` (xoá cả cache).

**Stage 3: foundation và zero-shot.**

```bash
PROFILE=smoke_30 EPOCHS=1 GPUS=0 bash scripts/diagnosis/foundation/ctfm_frozen.sh      # prognosis: scripts/prognosis/foundation/ctfm_frozen_{all,pe}.sh
CHECK_SLICE_ORDER=1 bash scripts/diagnosis/zero_shot/penet.sh      # bắt buộc chạy trước; sau đó SLICE_ORDER=<hướng thắng>
PROFILE=smoke_30 MAX_CASES=5 bash scripts/diagnosis/zero_shot/radar.sh
```

CT-FM wrapper (driver `scripts/tool/run_ctfm_frozen.sh`): `ACTION` = `prepare` | `train` | `evaluate` | `all` (mặc định) | `preflight` | `dry`. `prepare` chạy `tools/data/build_ctfm_cache.py` (một lần mỗi profile; `REBUILD_CACHE=1`, `VERIFY_CACHE=1`); `OVERWRITE=1` chỉ thay `epoch_<EPOCHS>/`. Bước đã xong được bỏ qua (`tools/run_status.py`). Bốn arm `foundation.ct_fm_frozen.anatomy_*` không có wrapper: `python run.py run <name> --gpus 0` (cần cache CT-FM và ROI). RADAR cần conda env riêng (`RADAR_PYTHON`), resume được bằng `RESUME=1`; PENet thì không ([`scripts/README.md`](../scripts/README.md)).

**Stage 4: diagnosis và baseline zoo.**

```bash
python run.py run diag.global.single --gpus 0
python run.py run diag.anatomy.concat --gpus 0,1
```

`run.py run` chỉ **train**. Metric test kèm CI do bước evaluate tạo ra; lặp lại đúng các `--set` đã dùng khi train:

```bash
python tools/launch.py --config configs/runs/02_diagnosis/anatomy/single_concat.yaml --evaluate --allow-full --gpus 0
# hoặc một tiến trình: python tools/tasks/evaluate.py --config <cùng config> --allow-full
```

Arm anatomy-aware trên profile khác `full_inspect` phải trỏ tới run ROI của profile đó (preset `configs/components/anatomy.yaml` cố định run ROI của `full_inspect`):

```bash
python run.py run diag.anatomy.concat --gpus 0 --set data.profile=test_500_sample \
  --set data.roi_manifest=roi/ROI_anatomy_and_controls__ds_test_500_sample/roi_manifest.csv
```

Baseline zoo (driver `scripts/tool/run_baseline_grid.sh`): `bash scripts/diagnosis/baselines/{prepare_weights,smoke,summarize}.sh`, `GPUS=0,1,2,3 bash scripts/diagnosis/baselines/exp01_baselines/run_all.sh`, hoặc một model: `PROFILE=smoke_30 EPOCHS=1 bash scripts/diagnosis/baselines/exp01_baselines/3D/resnet18_3d.sh`. Chi tiết: [`scripts/diagnosis/baselines/README.md`](../scripts/diagnosis/baselines/README.md).

**Stage 5: prognosis** (chỉ `prog.spesi`, `prog.image` ready).

```bash
python run.py run prog.image --gpus 0
# sPESI không train: chấm trên cùng danh sách ca (--restrict-to .../cache/full_inspect/clinical/spesi_evaluable.csv)
python tools/tasks/score_baseline.py --config configs/runs/03_prognosis/modality/spesi.yaml --score-column spesi --allow-full \n  --restrict-to ${PE_DERIVED_ROOT}/cache/full_inspect/clinical/spesi_evaluable.csv
```

**Stage 6: anatomy analysis.**

```bash
for V in global_only global_heart global_pa global_lung full_moe full_no_router; do python run.py run ablation.arch.$V --gpus 0; done
python run.py run anatomy.remove_pa --gpus 0 --patient-id <PATIENT_ID>   # smoke; thêm --allow-full cho toàn bộ
```

`anatomy.remove_*` đọc `lineage.source_checkpoint` = `${PE_CLOUD_ROOT}/pe-project/outputs/diagnosis/DX_anatomy_concat/epoch_50/checkpoint/best.ckpt`. Train `diag.anatomy.concat` với ngân sách epoch/profile khác thì phải `--set lineage.source_checkpoint=<path>` (và `data.roi_manifest` như trên).

**Tổng hợp:** `python tools/build_summary.py` (gom mọi `result.json`; output root lấy từ `PE_CLOUD_ROOT` hoặc `--output-root`) và `python tools/baselines/summarize.py --exp exp01_baselines --profile full_inspect`.

## 6. Song song và multi-GPU

Chạy đồng thời được miễn input đã có và **không hai job ghi cùng một run directory** (thứ tự phụ thuộc ở mục 4). Khác profile/backbone/init thì id khác (stamping); khác `seed` thì va collision; silver không stamp profile.

Nhiều experiment trên một máy nhiều GPU: `tools/launch_parallel.py --config sweep.yaml --dry-run`, với `sweep.yaml` có `parallel: {enabled, devices: [0,1,2,3], gpus_per_job, jobs: [{config: ...}]}` (mỗi job một config khác nhau; `args` chỉ nhận cờ scope và `--resume`, không nhận `--set`).

- Train/evaluate diagnosis, prognosis, ablation: DDP một process mỗi GPU, `--gpus 0,1` → `torchrun --standalone --nproc-per-node=2` (`tools/launch.py`, `source/distributed/launcher.py`); `CUDA_VISIBLE_DEVICES` là GPU vật lý, child nhận `compute.devices` logic `[0,1]`.
- Silver: chia report theo rank (mỗi rank load MedGemma, kiểm tra VRAM). Segmentation: chia study cho từng GPU, không DDP. ROI: CPU worker (`roi.workers`), `--gpus` bị bỏ qua.
- Baseline zoo: pool GPU qua `GPUS`, `JOBS_PER_GPU`, `GPUS_PER_JOB` (`scripts/tool/run_baseline_grid.sh`).

`GPUS=` rỗng (hoặc không truyền `--gpus`) chạy CPU.

## 7. Ba gốc output

```text
/mnt/Stanford_INSPECT_dataset/                 raw INSPECT, chỉ đọc
/mnt/pe-project/outputs/
├── derived/
│   ├── datasets/<profile>/                    cohort: manifest, data_quality.md, dataset.json
│   └── cache/<profile>/                       cache nặng: volume, CT-FM feature, clinical
└── pe-project/outputs/                        <output_root>: mọi run
    ├── segmentation/  roi/  silver_label/
    ├── diagnosis/  prognosis/  counterfactual/
    ├── ablation/{architecture,ehr}/
    ├── EDA/<profile>/                         analysis/run_eda.py
    └── summary/                               tools/build_summary.py
```

Thư mục `output/` ở gốc repo là bản sao cục bộ cùng layout. Đường dẫn một run: `<output_root>/<FAMILY_PATHS[experiment.family]>/<output_id hoặc id>[/epoch_<training.epochs>]` (`OutputManager.run_dir()`).

Family → thư mục: `segmentation`/`roi`/`silver` → `segmentation/`, `roi/`, `silver_label/`; `diagnosis`, `prognosis`, `counterfactual` cùng tên (diagnosis gồm cả zero-shot và baseline zoo `diagnosis/BASE/`); `architecture_ablation`/`ehr_ablation` → `ablation/architecture/`, `ablation/ehr/`. Run `01_foundation` nằm trong `diagnosis/`/`prognosis/`. Id có thể mang hậu tố stamp, ví dụ `diagnosis/DX_ctfm_frozen__ds_smoke_30/epoch_1/`.

## 8. Stage 0, segmentation, ROI, silver

```text
derived/datasets/<profile>/
├── data_quality.md          cohort funnel, nhãn/missing theo task, QC
├── dataset.json             provenance: profile, rule, sampling, hash, fingerprint tiền xử lý
├── logs.txt
└── manifests/
    ├── ctpa.csv  diagnosis.csv  prognosis*.csv  paired_reports.csv  reports.csv  exclusions.csv
    ├── ct_fm/*.csv          manifest trỏ tới CT-FM feature (cột pooled_path)
    └── experiments/...      manifest k-fold / fraction của baseline zoo

derived/cache/<profile>/
├── volumes/                 cache CT RAS đã tiền xử lý, dùng chung mọi stage
├── ct_fm/                   features/<study>.npy, pooled/<study>.npy
└── clinical/                ehr_features.csv, spesi_*.csv, modality_availability.csv, ...
```

Manifest luôn có `patient_id, study_id, split, image_path`; `split` ∈ `train | validation | test | external`. Không có cột `*_mask_path`: mask đọc từ run ROI. Các stage sinh dữ liệu dùng `prepare_resumable_run()` (không có `epoch_<E>/`, mặc định tiếp tục run dở):

| Run | File chính | QC / log |
|---|---|---|
| `segmentation/SEG_pseudo_anatomy[__ds_<p>]/` | `masks/<patient>/<study>/<anatomy>.nii.gz`, `manifest.parquet` | `previews/`, `qc_summary.csv`, `logs/run.log` |
| `roi/ROI_anatomy_and_controls[__ds_<p>]/` | `rois/<patient>/<study>/<roi>.nii.gz`, `roi_manifest.csv` | `previews/`, `logs/roi_qc.csv` |
| `silver_label/medgemma/` | `silver_labels.csv`, `silver_label_confidence.csv` | `logs/silver_label_qc.csv`; cache `.state/<report_hash>.json` |

Ý nghĩa cột: [01_data_pipeline.md](01_data_pipeline.md).

## 9. Run train + evaluate: `epoch_<E>/`

Áp dụng cho diagnosis, prognosis, ablation, CT-FM frozen, baseline zoo. Thư mục đặt theo **ngân sách** `training.epochs` (không phải số epoch thực chạy): `EPOCHS=1` và `EPOCHS=30` của cùng id là hai run độc lập.

```text
<family>/<id>/epoch_<E>/
├── resolved_config.yaml     config sau merge + --set + env + validate (có config_hash)
├── checkpoint/              best.ckpt (primary_val_metric cao nhất; mặc định −val_loss), last.ckpt
├── history.csv  training_curves.png   (train_task.py)
├── logs.txt                 log train, evaluate ghi nối tiếp
├── result.json              payload máy đọc (train tạo, evaluate cập nhật)
├── result.csv               bảng metric cho người đọc                  (evaluate.py)
├── predictions.csv          xác suất từng ca                           (evaluate.py)
├── reporting_checklist.json STARD-AI (diagnosis) / TRIPOD-AI (prognosis)
├── preview/                 Grad-CAM trên validation                   (evaluate.py)
└── smoke/<digest>/          evaluate giới hạn ca (--patient-id / --max-cases)
```

Sau evaluate toàn bộ, `cleanup_task_run()` xoá các bản trùng ở gốc bundle (`best.ckpt`, `config.yaml`, `metrics.json`, `logs/`, ...); run **chưa evaluate** còn các bản tạm này và chưa có `result.csv`.

**`result.csv`** (`write_result_csv()`): một dòng mỗi (target, split), cùng cột cho mọi run:

```text
experiment, target, split, n_patients, n_studies, n_pos, n_neg,
auroc, auroc_ci_low, auroc_ci_high, auprc, auprc_ci_low, auprc_ci_high,
threshold, threshold_rule, sensitivity, specificity, ppv, npv, f1,
balanced_accuracy, accuracy, brier, [calibration_intercept, calibration_slope], note
```

- `calibration_*` chỉ prognosis, cần ≥ `evaluation.calibration_min_events` (mặc định 10) ca mỗi lớp.
- `threshold` (`p ≥ threshold` → dương) chỉ ảnh hưởng sensitivity, specificity, PPV, NPV, F1, accuracy, balanced accuracy.
- `threshold_rule`: `youden_on_validation` (Youden J trên validation, áp cho mọi split); `default_0.5_validation_one_class` (validation một lớp, dùng 0.5); `locked_internal_validation` (external test dùng threshold khoá từ INSPECT validation).
- CI 95% là patient bootstrap, **chỉ ở dòng `test`**, chỉ cho AUROC / AUPRC; CI mọi metric nằm trong `result.json`.
- Ô trống = không tính được (không ghi `nan`); `note` ghi lý do (split một lớp, `n_pos`/`n_neg` < 5, thiếu sự kiện cho calibration).

**Các file khác:**

- `predictions.csv`: `split, target, patient_id, study_id, y_true, y_prob, y_pred`; dùng làm `--reference-predictions` cho paired bootstrap (evaluate ghi thêm `bootstrap_metrics.parquet` và khoá `paired_vs_reference`).
- `history.csv`: một dòng mỗi epoch (`train_loss`, `val_loss`, `primary_val_metric`, `lr`, `peak_vram_gb`, loss từng head, AUROC/AUPRC khi `training.record_epoch_auc` bật).
- `result.json` (`compact_result()`): `experiment`, `lineage` (backbone, init, source checkpoint + SHA-256, dataset, fusion, code commit), `data`, `model` (params, GFLOPs, kết quả load weight), `compute`, `evaluation.metrics.<tên> = {value, ci_low, ci_high}` + threshold + `training` (`epochs_run`, `stopped_early`), `evaluation_scope` (`full_test` hoặc smoke), `evaluation_checkpoint`, `config_hash`. `tools/build_summary.py` đọc file này; [04_experiments.md](04_experiments.md) lấy số từ `evaluation.metrics`.
- `preview/NN_<patient>_<study>_<TP|TN|FP|FN>.{html,png}` (`write_backbone_previews()`): Grad-CAM logit target chính trên **validation** (tối đa 5 study đầu; baseline zoo chọn 3 đúng + 3 sai tự tin nhất qua `preview:` trong `configs/components/baselines.yaml`). CAM không phải mask huyết khối. Tạo lại: `python tools/tasks/gradcam_preview.py --run-dir <epoch dir>`.

## 10. Layout đặc biệt và bảng tổng hợp

| Trường hợp | Thư mục | Nội dung |
|---|---|---|
| Evaluate smoke | `<id>/epoch_<E>/smoke/<digest>/` | `result.csv`, `predictions.csv`, `result.json`, `logs.txt` của tập ca giới hạn; không đụng kết quả full |
| sPESI score baseline | `prognosis/PR_spesi_only/score_baseline/` | `result.json`, `predictions.parquet`, `calibration_curve.parquet`; cạnh, không nằm trong, `epoch_<E>/` |
| Zero-shot / External test-only | `diagnosis/DX_zeroshot_{penet,radar}[__ds_<p>]/`, `diagnosis/<id>/` | Không có `epoch_<E>/`: `result.csv`, `predictions.csv`, `logs.txt`, `result.json`, `preview/` |
| Counterfactual | `counterfactual/CF_remove_{heart,pa,lung,random}/` | `counterfactual_predictions.parquet` (xác suất gốc vs sau khi xoá, theo cặp patient), `paired_bootstrap_metrics.parquet`, `result.json`; smoke: `CF_<x>__SMOKE_<digest>/` |
| Baseline zoo | `diagnosis/BASE/<profile>/<task>/runs/<model>__<head>__frac<PPP>/<fold>_seed<S>/epoch_<E>/` | Bundle như trên; mỗi `exp0*_*/` chứa bảng tổng hợp và symlink tới run dùng chung |

**`tools/build_summary.py`** quét mọi `result.json` (trừ `summary/`), ghi `<output_root>/summary/all_runs.csv` cùng `{silver,diagnosis,prognosis,architecture,ehr_ablation,counterfactual,roi,transfer}.csv` (family có run). **`tools/baselines/summarize.py --exp <exp>`** ghi `runs.csv`, `summary.csv`, `summary.md` (mean/std/n qua fold, seed) và hình `auroc_per_model.png` / `auroc_vs_fraction.png` / `mlp_vs_kan.png` (exp01/02/03) vào `diagnosis/BASE/<profile>/<task>/<exp>/`.

## 11. Resume và overwrite

| Stage | Mặc định khi thư mục đã có | `--resume` | `--overwrite` / `OVERWRITE=1` |
|---|---|---|---|
| Train | `OutputCollisionError` | Không hỗ trợ (`train_task.py` từ chối) | Xoá đúng `epoch_<E>/` rồi train lại; ngân sách epoch khác không bị ảnh hưởng |
| Evaluate | Dùng lại bundle đã train | — | Đánh giá lại |
| Segmentation, ROI | Tiếp tục run dở; từ chối nếu setting đổi | `run.py` không nhận | Xoá và chạy lại |
| Silver | Tiếp tục, dùng cache `.state/` | — | Xoá cả cache |
| Counterfactual | Full: collision; smoke: resume trong `__SMOKE_<digest>` | Có | Xoá và chạy lại |
| Dataset | `preprocessing.sh` bỏ qua profile đã đầy đủ cùng scope; dở hoặc khác scope thì dừng | — | Build lại |
| CT-FM wrapper | Bỏ qua bước đã xong; cùng ngân sách nhưng setting khác thì báo lỗi | — | Thay `epoch_<EPOCHS>/`; cache chỉ tính lại khi `REBUILD_CACHE=1` |

Thêm: run đã `completed` không thể `--resume`; `write_config()` so `config_hash` với `resolved_config.yaml` có sẵn, khác thì `OutputCollisionError` thay vì trộn hai config. Không xoá output cũ khi còn consumer (ví dụ `anatomy.remove_*` đọc checkpoint của `diag.anatomy.concat`).
