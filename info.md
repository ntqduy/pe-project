# PE Project — Tài liệu tổng hợp

> ⚠️ **LƯU Ý — repo đã được tái cấu trúc sau khi file này được viết.**
> Phần `source/` (mô tả từng module ở mục 11–12) vẫn đúng, nhưng **cấu trúc config, script và
> cách chạy đã thay đổi**:
> - Lệnh chạy chính bây giờ là `python run.py list | show | plan | preflight | dry | run <tên>`
>   với tên có nghĩa (`diag.anatomy.concat`, `prog.image_ehr_pesi`, `anatomy.remove_pa`...),
>   thay cho việc gọi trực tiếp `tools/launch.py --config configs/experiment/DX18....yaml`.
> - `configs/experiment/*.yaml` (flat) → `configs/runs/**` theo từng giai đoạn pipeline, cộng
>   `configs/components/**` chứa fragment tái sử dụng; toàn bộ config cũ được lưu trữ tại
>   `configs/legacy/experiment/`.
> - `scripts/**` (mỗi experiment một file .sh) → lưu trữ tại `scripts/legacy/`.
> - Danh mục experiment do con người đọc: `configs/experiments.yaml`.
> - Tài liệu hiện hành: `docs/PIPELINE.md`, `docs/EXPERIMENT_MAP.md`, `docs/BLOCKERS.md`,
>   `README.md`, `configs/README.md`, `tools/README.md`.
> - Repo **không có** thư mục `tests/` và không dùng pytest.
> - `docs/anatomy_aware_gap_analysis.md` và `docs/anatomy_aware_implementation.md` đã bị
>   xoá; nội dung anatomy-aware hiện nằm trong `README.md` (mục *Architecture*) và
>   `docs/PIPELINE.md`. Mọi tham chiếu tới hai file đó ở phía dưới chỉ còn giá trị lịch sử.
> - Mask giải phẫu **không** đến từ cột `*_mask_path` trong manifest nữa: chúng được đọc từ
>   ROI run qua `data.roi_manifest` + `data.roi_mask_ids`
>   (`configs/components/data/anatomy_masks.yaml`).
>
> File này tổng hợp toàn bộ thông tin về dự án: dự án làm gì, cấu trúc thư mục, ý nghĩa từng
> file/module, cách cài đặt, cách chạy, và các phần còn dang dở. Nội dung được tổng hợp từ
> `README.md`, `configs/README.md`, `scripts/README.md`, `tools/README.md`, `third_party/README.md`,
> `docs/*.md` và việc đọc trực tiếp mã nguồn trong `source/` và `tools/`.

## 1. Dự án này là gì?

`pe-project` là một **framework nghiên cứu** (không phải sản phẩm) cho bài toán **representation
learning trên ảnh CTPA 3D (CT Pulmonary Angiography)** phục vụ chẩn đoán và tiên lượng **thuyên tắc
phổi (Pulmonary Embolism – PE)**. Trọng tâm khoa học là kiến trúc "anatomy-aware": biểu diễn ảnh được
tách theo vùng giải phẫu (tim, động mạch phổi PA, phổi) rồi mới fusion lại, để có thể trả lời hai câu
hỏi:

- **Necessity** (tính cần thiết): nếu xoá vùng ROI đó khỏi input của một model đã train xong thì độ
  chính xác giảm bao nhiêu? (counterfactual, không train lại — nhóm `CF*`).
- **Sufficiency** (tính đủ): nếu chỉ train model bằng riêng vùng ROI đó (student) thì đạt được bao
  nhiêu phần trăm hiệu năng so với model dùng toàn bộ ảnh? (nhóm `RS*`, `KD*`).

Dự án có 3 task lâm sàng chính: **Diagnosis** (chẩn đoán PE và các đặc điểm đi kèm), **Prognosis**
(tiên lượng tử vong, kết hợp ảnh + lâm sàng + điểm PESI), và **Contour** (phân đoạn/tinh chỉnh vùng
huyết khối). Mỗi giai đoạn train sinh ra một checkpoint có "lineage" (phả hệ) tường minh, được kiểm
tra bằng bước `preflight` trước khi chạy.

Repo **chỉ dùng model/pipeline thật, không có code giả lập (mock)**. Rất nhiều giá trị cấu hình
(feature dimension, factory import path, tên cột dữ liệu thật...) được cố ý để trống làm
**placeholder chặn (blocker)** — xem mục 8 — để tránh việc "đoán" ra một pipeline chạy được nhưng sai
khoa học.

### Trạng thái hiện tại (quan trọng)

- Thư mục `tools/` (CLI) **đã tồn tại đầy đủ** và implement thật (không phải chỉ là tài liệu suông).
- Thư mục `tests/` được nhắc tới trong `docs/anatomy_aware_implementation.md` ("Implemented") nhưng
  **hiện KHÔNG tồn tại** trong repo — cần tạo lại nếu muốn chạy `pytest -q` như README mô tả.
- `third_party/repos/` đã có sẵn clone của `TotalSegmentator`, `lungmask`, `CT-CLIP`, `CT-FM`,
  `TotalFM` (xem mục 7.5), nhưng `third_party/versions.yaml` còn nhiều `checksum_sha256: null` và
  `third_party/weights/` chưa có model weight thật (bị `.gitignore`).
- Nhiều config backbone (`configs/backbone/*.yaml`) còn `factory`, `output_adapter`, `feature_dim`
  rỗng — đây là placeholder cố ý, `preflight` sẽ FAIL cho đến khi điền đúng theo checkpoint thật.

## 2. Pipeline nghiên cứu tổng thể

### 2.1. Support artifacts (segmentation / ROI / silver label)

```text
CTPA manifest
    |
    +--> SEG01: TotalSegmentator --> anatomy masks --> QC
    |                                  |
    |                                  +--> LungMask chỉ đối chiếu lung Dice
    |
    +--> ROI01: đọc SEG01 đã lưu --> ROI1–ROI8 --> QC

Report table
    |
    +--> SL00: MedGemma
    +--> SL01: rules + Falcon
    +--> SL02: rules + Falcon + MedGemma
                         |
                         +--> accepted | abstained | no_result
```

Ba nhánh này độc lập: ROI không tự chạy segmentation (chỉ đọc `SEG01/manifest.parquet`); silver label
chỉ đọc report, không phụ thuộc mask/ROI.

### 2.2. Shared encoder và downstream tasks

```text
Public CT checkpoint
    |
    +--> F01/F02/F03: load, chuẩn hoá checkpoint, profile; KHÔNG train
    |
    +--> D00 (none) hoặc D01 (MAE) / D02 (DINO) / D03 (SimCLR) / D04 (Anatomy-DAPT)
                    |
                    +--> AL01 image-report alignment
                              |
                              +--> C0 (chưa học từ silver labels)
                                      |
                                      +--> SE01 + accepted SL00/SL01/SL02
                                                |
                                                +--> C_silver

C0 hoặc C_silver --> Diagnosis / Prognosis / Contour (experiment riêng)
```

`C0` và `C_silver` là hai encoder song song để so sánh có/không silver supervision; khi so sánh, mọi
thứ khác (split, label, kiến trúc, hyperparameters) phải giữ nguyên, chỉ đổi
`lineage.source_checkpoint` của `image_encoder`.

### 2.3. Anatomy-aware diagnosis, necessity và sufficiency

```text
C0 hoặc C_silver -> spatial feature map từ full CTPA
                   |-> global average pool -> global adapter
                   |-> heart masked pool   -> heart adapter -> heart auxiliary heads
                   |-> PA masked pool      -> PA adapter    -> PA auxiliary heads
                   `-> lung masked pool    -> lung adapter  -> lung auxiliary heads

adapted global + heart + PA + lung -> concat+MLP hoặc soft-MoE -> PE +/-

trained full model + original masks -> CF01-CF04 remove-ROI inference (KHÔNG train lại)
C0 + ROI2/ROI4/ROI6/ROI8 input      -> RS06-RS09 hoặc KD01-KD04 (student, có/không distillation)
```

Lưu ý: vector heart/PA/lung là biểu diễn **pool theo vùng từ feature map của toàn bộ CTPA**, không
phải "chỉ nhìn thấy organ đó" theo nghĩa nghiêm ngặt — chỉ các arm "strict"/"student" (dùng input đã
bị che ROI) mới thực sự organ-only.

### 2.4. Bảng: mỗi stage tạo ra gì?

| Stage | Config chính | Input | Kết quả |
|---|---|---|---|
| Segmentation | `SEG01.yaml` | CTPA manifest | anatomy masks, manifest, QC, preview |
| ROI | `ROI01.yaml` | run SEG01 đã hoàn tất | ROI1–ROI8, manifest, QC, preview |
| Silver generation | `SL00/01/02.yaml` | report table | labels, QC, state theo report |
| Foundation | `F01/02/03.yaml` | public checkpoint | checkpoint format của project (không train) |
| DAPT | `D00–D04.yaml` | CTPA (D04 thêm PA mask) | adapted encoder |
| Alignment | `AL01.yaml` | CTPA + report embedding | encoder **C0** |
| Silver encoder | `SE01.yaml` | C0 + accepted silver targets | encoder **C_silver** |
| Diagnosis | `DX*.yaml` | CTPA + anatomy masks + PE labels | checkpoint + metrics |
| Prognosis | `PR*.yaml` | image + EHR + PESI | checkpoint + metrics |
| Contour | `CT*.yaml` | CTPA + embolus mask | checkpoint + metrics |
| Counterfactual | `CF01–CF04.yaml` | trained full model + CTPA + original masks | paired probability deltas; **không train** |
| ROI students | `RS06–RS09` / `KD01–KD04.yaml` | C0 + ROI01 + PE labels | student checkpoint độc lập + metrics |
| CTPA seg validation | `SEG02_CTPA_FINETUNE.yaml` | expert mask subset | **unavailable** đến khi có annotation contract |

Không script nào tự chạy prerequisite (ví dụ `DX01_train.sh` không tự tạo `SEG01`, `SL02`, `D02`,
`AL01`); `preflight` sẽ dừng sớm nếu artifact mà config yêu cầu chưa tồn tại.

## 3. Cấu trúc thư mục cấp cao

```text
pe-project/
├── configs/            YAML cấu hình, kế thừa qua `_base_`
│   ├── experiment/       config runnable (có experiment.id + stage): SEG01, ROI01, SL*, F*, D*,
│   │                     AL01, SE01, DX*, PR*, CT*, các ablation A/RM/SA/TR/CF/RS/KD
│   ├── backbone/         contract tích hợp CT-FM / CT-CLIP / TotalFM
│   ├── compute/          GPU mặc định hoặc CPU
│   ├── dapt/             fragment objective DAPT (none/MAE/DINO/SimCLR/Anatomy)
│   ├── alignment/        fragment objective image-report alignment
│   ├── task/             fragment field cho diagnosis/prognosis/contour
│   ├── silver/           fragment provider/model cho SL00/SL01/SL02
│   ├── parallel/         config scheduler nhiều job song song (không phải experiment config)
│   └── paths.yaml        data/output roots
├── source/             thư viện lõi: model, data, training engine, QC, pipeline logic (xem mục 7.2)
├── tools/              CLI Python theo domain — entrypoint thật, gọi vào source/ (xem mục 7.3)
├── scripts/             Bash wrapper mỏng quanh tools/ (xem mục 7.4)
├── docs/                ghi chú thiết kế/gap-analysis cho phần anatomy-aware
├── paper/               tài liệu tham khảo (PDF)
├── cache/               cache local, không commit (rỗng/không tồn tại mặc định)
├── third_party/
│   ├── repos/            clone upstream (TotalSegmentator, lungmask, CT-CLIP, CT-FM, TotalFM);
│   │                     không sửa bằng logic của project
│   ├── weights/          model weights local, không commit
│   ├── versions.yaml     URL/commit/checksum/status của từng dependency
│   └── README.md
├── pyproject.toml       package metadata + optional-dependency groups
├── requirements.txt     pin version cụ thể cho toàn bộ dependency
└── README.md
```

`tools/` không có wrapper Python trùng chức năng; logic mô hình/khoa học nằm hết trong `source/`,
`tools/` chỉ là orchestration (đọc config, gọi vào `source/`, ghi kết quả).

## 4. Cài đặt môi trường

Khuyến nghị Python 3.11 (yêu cầu `>=3.11,<3.15` theo `pyproject.toml`). Từ project root:

```bash
conda create -n pe311 python=3.11 -y
conda activate pe311
python -m pip install --upgrade pip setuptools wheel
```

Cài PyTorch phù hợp CUDA/driver của máy trước, rồi:

```bash
pip install -r requirements.txt
pip install -e .
```

Support pipeline (segmentation) còn cần cài CLI upstream tương ứng:

```bash
pip install -e third_party/repos/TotalSegmentator
pip install -e third_party/repos/lungmask
```

Không cài toàn bộ requirements của mọi upstream repo vào cùng môi trường nếu không cần — third-party
source/weight được quản lý theo `third_party/README.md` (mục 7.5).

**Nhóm dependency** (`pyproject.toml`):

| Nhóm | Gồm | Dùng cho |
|---|---|---|
| core (bắt buộc) | numpy, pandas, pyarrow, PyYAML, scikit-learn, scipy, torch, nibabel, matplotlib | mọi stage |
| `training` | torchvision, SimpleITK, monai, transformers, accelerate, huggingface-hub, safetensors, einops, sentencepiece | train model, silver LLM |
| `segmentation` | pydicom, scikit-image, fill-voids, more-itertools, tqdm | TotalSegmentator/LungMask |
| `dev` | pytest, pytest-cov, ruff | test và lint |

## 5. Dữ liệu và biến môi trường

```powershell
$env:PE_CLOUD_ROOT = "E:\PE_NU"
$env:PE_LOCAL_CACHE_ROOT = "E:\PE_NU\cache"
cd E:\PE_NU\source\pe-project
```

Với cấu hình mặc định (`configs/paths.yaml`), project resolve:

```text
${PE_CLOUD_ROOT}/data/Stanford_INSPECT_dataset   raw, chỉ đọc
${PE_CLOUD_ROOT}/data/derived                    manifest và dữ liệu đã chuẩn hoá
${PE_CLOUD_ROOT}/pe-project/outputs              mọi output experiment
```

Manifest nằm dưới `data/derived/manifests/`, tối thiểu phải có cột `patient_id, study_id, split,
image_path`. `split` chỉ nhận `train | validation | test | external`; một patient không được xuất
hiện ở nhiều split. `reports.parquet` cho silver generation cần `patient_id, study_id, report_id,
report_text`.

Nếu dữ liệu chưa có split, tạo đúng một patient-level split tường minh (không tạo split ngầm trong
train/evaluate, không dùng test split để chọn checkpoint):

```bash
python tools/data/create_split.py --input SOURCE.csv --output manifests/ctpa.csv --create-split --seed 42
```

## 6. `preflight` — kiểm tra trước khi chạy

`preflight` không phải một model stage, mà là bước kiểm tra cấu hình: resolve YAML inheritance/env
var/GPU override; kiểm tra manifest, split theo patient, duplicate ID, file bắt buộc; kiểm tra upstream
repo/checkpoint/provider/supervision artifact; kiểm tra GPU yêu cầu có tồn tại; kiểm tra output root
ghi được và phát hiện collision. Nó **không** chạy inference/train, không kiểm tra chất lượng lâm sàng,
không thay thế QC sau stage. `launch.py` tự gọi preflight; chạy riêng để sửa cấu hình trước khi launch
thật:

```bash
python tools/preflight.py --config configs/experiment/SEG01.yaml --gpus 0
python tools/preflight.py --config configs/experiment/DX01.yaml --gpus 0,1
```

Nếu output đã tồn tại: thêm `--resume` (khi stage hỗ trợ) hoặc `--overwrite` (chạy lại chủ động).

## 7. Cách chạy

### 7.1. Ba chế độ chọn dữ liệu (loại trừ nhau)

| Chế độ | Cờ | Ghi chú |
|---|---|---|
| Một/nhiều patient | `--patient-id PATIENT_ID` (lặp lại được) | mọi study/report của patient đó |
| N item đầu | `--max-cases N` (segmentation/ROI) hoặc `--max-reports N` (silver) | kiểm tra nhanh |
| Toàn bộ | `--allow-full` | phải bật tường minh |

Không đặt đồng thời `PATIENT_ID` và `ALLOW_FULL` ở wrapper Bash; wrapper ưu tiên `PATIENT_ID`.

**Một patient:**

```bash
python tools/create_masks/generate_masks.py --config configs/experiment/SEG01.yaml --patient-id PATIENT_001 --gpus 0
python tools/build_rois/build_rois.py --config configs/experiment/ROI01.yaml --patient-id PATIENT_001
python tools/launch.py --config configs/experiment/SL02.yaml --gpus 0 --patient-id PATIENT_001
```

ROI chỉ chạy được sau khi SEG01 đã tạo mask của patient đó.

**Một item đầu tiên:**

```bash
python tools/create_masks/generate_masks.py --config configs/experiment/SEG01.yaml --max-cases 1 --gpus 0
python tools/build_rois/build_rois.py --config configs/experiment/ROI01.yaml --max-cases 1
python tools/launch.py --config configs/experiment/SL02.yaml --gpus 0 --max-reports 1
```

**Full input:**

```bash
python tools/create_masks/generate_masks.py --config configs/experiment/SEG01.yaml --allow-full --gpus 0,1
python tools/build_rois/build_rois.py --config configs/experiment/ROI01.yaml --allow-full
python tools/launch.py --config configs/experiment/SL02.yaml --gpus 0,1 --allow-full
```

### 7.2. Chạy từng bước bằng script Bash (Git Bash / WSL / Linux)

| Bước | Lệnh | Ghi chú |
|---|---|---|
| 1. Segmentation | `bash scripts/segmentation/SEG01_generate_masks.sh` | mặc định 1 case; `PATIENT_ID=`, `MAX_CASES=`, `ALLOW_FULL=1` + `GPUS=` |
| 2. ROI1–ROI8 | `bash scripts/roi/ROI01_build.sh` | CPU song song theo `roi.workers` trong `ROI01.yaml` |
| 3. Silver labels | `bash scripts/silver/SL02_hybrid.sh` (hoặc `SL00_medgemma_only.sh` / `SL01_rules_falcon.sh`) | |
| 4. Foundation checkpoint | `bash scripts/foundation/F01_ct_fm.sh 0` (hoặc `F02_ct_clip.sh` / `F03_totalfm.sh`) | chỉ CPU hoặc 1 GPU, không train |
| 5. DAPT | `bash scripts/dapt/D02_dino.sh 0,1` | lựa chọn: `D00_none`, `D01_mae`, `D02_dino`, `D03_simclr`, `D04_anatomy` |
| 6. Alignment → C0 | `bash scripts/alignment/AL01_image_report.sh 0,1` | |
| 7. Silver-supervised → C_silver | `bash scripts/silver_encoder/SE01_train.sh 0,1` | đổi nguồn silver: `--set silver_training.silver_source=SL00` |
| 8. Downstream | `bash scripts/diagnosis/DX01_train.sh 0,1`, `scripts/prognosis/PR01_train.sh`, `scripts/contour/CT01_train.sh` | wrapper train xong tự gọi evaluate |

Gọi evaluate riêng:

```bash
python tools/tasks/evaluate.py --config configs/experiment/DX01.yaml --gpus 0,1
```

### 7.3. Ví dụ nhóm anatomy-aware (necessity/sufficiency)

```bash
# concat+MLP reference (clean, không cần router)
python tools/launch.py --config configs/experiment/DX18_ANATOMY_FULL.yaml --gpus 0,1

# multitask native-branch vs accepted-silver-branch
python tools/launch.py --config configs/experiment/DX16_MULTITASK_NATIVE.yaml --gpus 0,1
python tools/launch.py --config configs/experiment/DX17_MULTITASK_SILVER.yaml --gpus 0,1

# necessity: frozen inference, KHÔNG retrain
python tools/launch.py --config configs/experiment/CF02_REMOVE_PA.yaml --gpus 0,1 --allow-full

# sufficiency: cùng ROI, không KD và có KD
python tools/launch.py --config configs/experiment/RS07_PA_GT.yaml --gpus 0,1
python tools/launch.py --config configs/experiment/KD02_PA.yaml --gpus 0,1

# prognosis baseline PESI-only vs multimodal anatomy+clinical+PESI (cùng cohort confirmed_acute_pe)
python tools/launch.py --config configs/experiment/PR26_PESI_ONLY.yaml --gpus 0
python tools/launch.py --config configs/experiment/PR22_ANATOMY_CLINICAL_PESI.yaml --gpus 0,1
```

Smoke inference trên một patient thật ghi vào run riêng, **không đè** full evaluation:

```bash
python tools/launch.py --config configs/experiment/CF02_REMOVE_PA.yaml --gpus 0 --patient-id PATIENT_001
python tools/tasks/evaluate.py --config configs/experiment/DX18_ANATOMY_FULL.yaml --gpus 0 --patient-id PATIENT_001
```

### 7.4. GPU và chạy song song

- Train dùng PyTorch DDP, mỗi GPU một process (`launch.py` set `CUDA_VISIBLE_DEVICES`, remap về ID
  logic `0..N-1`, gọi `torchrun` khi ≥2 GPU).
- Silver generation shard report giữa các process; mỗi rank giữ model trên GPU local (SL02 phải chứa
  cả Falcon và MedGemma trên mỗi rank — kiểm tra VRAM trước khi chạy full).
- Segmentation shard theo study bằng `--gpus`, không đi qua DDP.
- ROI dùng CPU workers (`roi.workers`), không dùng GPU launcher.

**Một job nhiều GPU:**

```bash
python tools/launch.py --config configs/experiment/D02.yaml --gpus 0,1,2,3
```

**Nhiều job độc lập cùng lúc** — sửa `configs/parallel/experiments.yaml` (`parallel.enabled: true`,
chọn `devices`, `gpus_per_job`, danh sách `jobs`), rồi:

```bash
python tools/launch_parallel.py --config configs/parallel/experiments.yaml --dry-run
python tools/launch_parallel.py --config configs/parallel/experiments.yaml
# hoặc wrapper tương đương
bash scripts/parallel/run_batch.sh --dry-run
```

`gpus_per_job: 1` trên `[0,1,2,3]` → tối đa 4 job/wave; `gpus_per_job: 2` → 2 job/wave, mỗi job dùng
2 GPU nội bộ (distributed). Không xếp hai stage có quan hệ phụ thuộc vào cùng một wave (ví dụ AL01
không nên chạy cùng lúc với D02 nếu đang chờ chính checkpoint D02 đó).

### 7.5. Override, resume, tổng hợp kết quả

```bash
python tools/launch.py --config configs/experiment/SE01.yaml --set silver_training.silver_source=SL00 --gpus 0,1
python tools/launch.py --config configs/experiment/D02.yaml --gpus 0,1 --dry-run   # chỉ in command sau preflight
python tools/data/create_split.py --input SOURCE.csv --output TARGET.csv --create-split --seed 42
python tools/build_summary.py                                                        # gộp result.json đã có
python tools/sync_project.py --help                                                  # sync source sang cloud
pytest -q                                                                             # unit test (hiện chưa có thư mục tests/)
```

## 8. Trạng thái config: những gì còn phải điền trước khi chạy thật

Các placeholder sau được giữ **cố ý** để tránh bịa ra một contract không khớp checkpoint/dataset thật
— `preflight` sẽ FAIL cho đến khi hoàn chỉnh:

1. `configs/backbone/*.yaml` (ct_fm, ct_clip, totalfm): `factory`, `output_adapter`, `feature_dim` và
   checksum đúng checkpoint đang trống.
2. `configs/alignment/image_report.yaml`: `report_embedding_columns` đang rỗng (cũng dùng bởi
   `DX04_report_only`).
3. `configs/experiment/PR01.yaml`: `ehr_columns`/`ehr_columns_full`/`ehr_columns_common` rỗng trong
   khi `ehr_input_dim: 32`; `ehr_columns_common` phải là tập con của `ehr_columns_full`.
4. `configs/experiment/DX06_rsna_single.yaml`: manifest RSNA (`manifests/rsna_diagnosis.csv`) là
   placeholder.
5. `configs/experiment/PR16_allcomer.yaml`: manifest all-comer (`manifests/prognosis_allcomer.csv`)
   là placeholder, tách biệt với cohort acute-PE chính (`PR18`–`PR27`).
6. `configs/experiment/PR17_concept_bottleneck.yaml`: mọi concept phải trỏ cột nhãn thật
   (native/silver đã validate/expert-reviewed) trước khi bật `concept_bottleneck.enabled: true`.
7. `third_party/versions.yaml`: nhiều `checksum_sha256` vẫn `null`, cần điền sau khi có weight thật.
8. Cần đặt đầy đủ upstream weights theo `third_party/weights/README.md`, và xác nhận mọi
   path/cột mask trong manifest tồn tại.
9. Cần tạo đủ `ROI01` cho `ROI2/ROI4/ROI6/ROI8` trước khi chạy các arm strict-organ/ROI-student.
10. `SEG02_CTPA_FINETUNE` cố ý **unavailable** cho đến khi có expert annotation + factory/checkpoint
    đã được kiểm tra (xem `docs/anatomy_aware_gap_analysis.md`).

Không được đoán các giá trị này — chỉ điền sau khi kiểm tra checkpoint factory, output shape và schema
manifest thật.

## 9. Quy tắc vận hành & checklist

- Không ghi output nghiên cứu vào source tree; output phải nằm dưới project output root
  (`${PE_CLOUD_ROOT}/pe-project/outputs`).
- Không dùng cùng `experiment.id` cho hai cấu hình khoa học khác nhau.
- Dùng `--resume` chỉ khi stage hỗ trợ; dùng `--overwrite` khi chủ động xoá run cũ và chạy lại.
- Luôn xem QC của SEG/ROI/SL trước khi train stage phụ thuộc.
- Giữ `C0` và `C_silver` thành hai checkpoint độc lập.
- Chạy `tools/build_summary.py` để tổng hợp các `result.json` đã có.

---

## 10. Ý nghĩa `configs/`

Chỉ file trong `configs/experiment/` có `experiment.id` + `experiment.stage` mới runnable trực tiếp
qua `tools/launch.py`. Các folder khác (`backbone/`, `compute/`, `dapt/`, `alignment/`, `task/`,
`silver/`) chỉ là **fragment** được kế thừa bằng khoá `_base_`, không đưa thẳng vào launcher.

### 10.1. Cơ chế kế thừa (`_base_`)

Ví dụ `D02.yaml` kế thừa `D01` rồi thay objective bằng DINO:

```yaml
_base_:
  - D01.yaml
  - ../dapt/dino.yaml
experiment:
  id: D02
  stage: dapt
```

Base được merge theo thứ tự, mapping sau ghi đè mapping trước; list/scalar bị thay toàn bộ (không nối
ngầm); đường dẫn base resolve tương đối theo file chứa `_base_`. Override nhỏ ở CLI dùng `--set
KEY=VALUE` (xem mục 7.5) — nếu thay đổi đủ lớn để thành một experiment khoa học khác, tạo YAML +
`experiment.id` mới thay vì dựa vào chuỗi override khó truy vết.

### 10.2. Nhóm prefix `experiment.id`

| Prefix | Ý nghĩa |
|---|---|
| `SEG` | pseudo-anatomy segmentation (TotalSegmentator + LungMask) |
| `ROI` | ROI construction (ROI1–ROI8) |
| `SL` | silver-label generation (SL00 MedGemma-only, SL01 rules+Falcon, SL02 hybrid/adjudicated) |
| `F` | foundation checkpoint materialization (chỉ load/profile, không train) |
| `D` | DAPT (D00 none, D01 MAE, D02 DINO, D03 SimCLR, D04 Anatomy-DAPT) |
| `AL` | image-report alignment, tạo **C0** |
| `SE` | silver encoder adaptation, tạo **C_silver** |
| `DX` | diagnosis |
| `PR` | prognosis |
| `CT` | contour |
| `A`, `RM`, `SA`, `TR` | ablation/counterfactual/transfer đời cũ (legacy) |
| `CF` | frozen counterfactual inference (necessity, không train) |
| `RS`, `KD` | ROI-only student (RS: ground-truth loss; KD: có thêm distillation từ teacher đóng băng) |

Chi tiết một số nhóm mở rộng cho anatomy-aware:

- `DX14_FULL_SCRATCH` / `DX15_FULL_FM`: global full-CT, khởi tạo random vs public FM.
- `DX16_MULTITASK_NATIVE` / `DX17_MULTITASK_SILVER`: auxiliary head gắn trực tiếp vào branch
  heart/PA/lung, dùng nhãn native hay silver.
- `DX18_ANATOMY_FULL`: concat+MLP reference, không bắt buộc router — **cần train trước** khi chạy
  `CF*`/`KD*`.
- `DX19_HEART_ONLY` … `DX22_RANDOM_ONLY`: strict/matched-control diagnosis arms, đọc ROI2/ROI4/ROI6
  từ `ROI01/manifest.parquet`.
- `CF01_REMOVE_HEART` … `CF04_REMOVE_RANDOM`: stage `counterfactual` — chỉ frozen inference trên cùng
  full model + mask gốc; **không** phải training/ablation launcher ngầm.
- `RS06_HEART_GT` … `RS09_RANDOM_GT`: ROI-only student từ C0, ground-truth loss.
- `KD01_HEART` … `KD04_RANDOM`: student tương ứng cộng thêm frozen-teacher knowledge distillation.
- `PR18_IMAGE_FULL` … `PR27_CLINICAL_ONLY`: mười arm PESI-only/clinical-only/global-image/multimodal
  /anatomy-aware/organ-only prognosis trên cùng cohort `confirmed_acute_pe` (tách biệt khỏi `PR01` và
  `PR16_allcomer`).
- `SEG02_CTPA_FINETUNE`: contract tùy chọn, cố ý FAIL preflight vì chưa có expert mask/factory.

`task.auxiliary_targets` có schema `organ -> target -> {type, classes?, source}` với `source` chỉ
nhận `native | silver | expert_reviewed`; không target nào được tự suy ra từ tên cột.
`evaluation.split_strategy` nhận `patient_holdout | temporal_holdout | patient_stratified_cv`.

### 10.3. Compute/GPU

`configs/compute/default.yaml` dùng 1 GPU mặc định (`strategy: auto`, `devices: [0]`); `--gpus` ở CLI
ghi đè `devices`/`strategy`/`accelerator`. `configs/compute/cpu.yaml` dành cho ROI (song song bằng
`roi.workers`, không DDP).

### 10.4. `configs/parallel/experiments.yaml`

Không phải experiment config mà là config cho **scheduler**: `devices` (GPU pool vật lý),
`gpus_per_job` (kích thước mỗi group), `jobs` (danh sách config + `args` tùy chọn — chỉ dùng cho
silver selection hoặc `--resume`). Có thể trộn training và silver generation nếu không có
prerequisite đang chờ nhau.

### 10.5. Danh mục file trong `configs/`

- `paths.yaml` — khai báo `cloud_root`, `data_root`, `output_root`... (đọc từ biến môi trường
  `PE_CLOUD_ROOT`).
- `backbone/ct_fm.yaml`, `ct_clip.yaml`, `totalfm.yaml` — contract tích hợp từng public image
  backbone (`repo`, `checkpoint`, `factory`, `output_adapter`, `feature_dim`, `strict_load`).
- `compute/default.yaml`, `compute/cpu.yaml` — chiến lược compute mặc định/CPU.
- `dapt/none.yaml`, `mae.yaml`, `dino.yaml`, `simclr.yaml`, `anatomy_dapt.yaml` — fragment objective
  DAPT tương ứng D00–D04.
- `alignment/image_report.yaml` — fragment objective alignment ảnh–report (dùng cho AL01).
- `task/diagnosis.yaml`, `prognosis.yaml`, `contour.yaml` — field mô hình/task chung (targets, kiến
  trúc fusion, expert_dim...) được các experiment `DX*/PR*/CT*` kế thừa.
- `silver/hybrid.yaml`, `rules_falcon.yaml`, `medgemma_only.yaml` — cấu hình provider/model cho
  SL02/SL01/SL00.
- `parallel/experiments.yaml` — config scheduler song song (mục 10.4).
- `experiment/*.yaml` — 121 file config runnable, đặt tên theo prefix ở bảng 10.2 (SEG01, ROI01,
  SL00–SL02, F01–F03, D00–D04, AL01, SE01, SE02_from_rsna, DX01–DX22, PR01–PR27, CT01–CT02, các
  ablation A01–A08, RM01–RM04, SA00–SA03, TR01, RS01–RS09, KD01–KD04, CF01–CF04, SEG02).

---

## 11. Ý nghĩa `source/` (thư viện lõi)

`source/` là thư viện thuần (ít side-effect): I/O dữ liệu, encoder/adapter/fusion/PEFT, model cho
từng task, training/checkpoint/output engine, và các tiện ích metrics/imaging/distributed/utils dùng
chung. `source/__init__.py` chỉ khai báo `__version__ = "0.1.0"`.

### 11.1. `source/clinical/` — dữ liệu lâm sàng (EHR, PESI)

- `encoder.py` — `ClinicalEncoder(nn.Module)`: MLP nhỏ trên giá trị lâm sàng thô, có impute/normalize
  fit trên train và cờ missingness tùy chọn (bọc `ClinicalPreprocessor`).
- `pesi.py` — `PESIResult` (dataclass) + `compute_pesi`/`compute_spesi`: cài đặt thuần Python công
  thức điểm **PESI** và **sPESI** (độ nặng thuyên tắc phổi) từ hồ sơ lâm sàng.
- `preprocessing.py` — `ClinicalPreprocessor(nn.Module)`: fit impute (mean/median/constant) và
  chuẩn hoá z-score **chỉ trên tập train**, lưu như buffer checkpoint, có `export_state`/`import_state`.
- `__init__.py` — re-export `ClinicalEncoder`, `ClinicalPreprocessor`, `PESIResult`, `compute_pesi`,
  `compute_spesi`.

### 11.2. `source/components/` — khối xây dựng model dùng chung

- `targets.py` — `TargetSpec` (binary/multiclass/regression); `masked_target_loss`/
  `masked_multitask_loss` tính loss theo từng target có mask (BCE/CE/regression), xử lý batch rỗng.
- `anatomy.py` — `AnatomyFeatureOutput` + `extract_anatomy_features(...)`: encode CTPA một lần, pool
  đặc trưng global/organ (tim/PA/phổi) qua ROI extractor, rồi adapt qua `OrganAdapterBank`.
- `__init__.py` — re-export `OrganAdapterBank`, `TargetSpec`, `normalize_target_specs`.

**`components/adapters/`** — adapter chiếu đặc trưng encoder đã pool về không gian expert chung:
`BottleneckMLPAdapter`, `ResidualAdapter`, `LoRAFeatureAdapter`, factory `build_organ_adapter`, và
`OrganAdapterBank(nn.Module)` quản lý các adapter global/heart/PA/lung độc lập kèm mask khả dụng theo
từng sample (`organ.py`).

**`components/encoders/`** (không phải ảnh):
- `ehr.py` — `EHREncoder(ClinicalEncoder)`: alias tương thích ngược.
- `pesi.py` — `PESIEncoder(nn.Module)`: MLP nhỏ trên điểm PESI/sPESI, có xử lý missingness.

**`components/encoders/image/`** — contract cho image backbone:
- `base.py` — `ImageFeatures` (dataclass) và `BaseImageEncoder(nn.Module, ABC)` — interface ổn định
  (`forward_features`, `get_feature_map`, `get_global_embedding`, `load_pretrained_weights`) bọc
  quanh các encoder CT bên thứ ba đã được kiểm tra.
- `external.py` — `InspectedExternalEncoder` + `build_inspected_external(...)`: adapter tổng quát nạp
  bất kỳ model bên thứ ba nào qua cặp `factory`/`output_adapter` đã cấu hình + checkpoint, ép theo
  đúng contract `ImageFeatures`.
- `ct_clip.py`, `ct_fm.py`, `totalfm.py` — wrapper mỏng `build_ct_clip`/`build_ct_fm`/`build_totalfm`
  gọi `build_inspected_external` cho từng backbone tương ứng.
- `registry.py` — `register_backbone`, `build_image_encoder`, `registered_backbones`: registry theo
  tên (`ct_fm`, `ct_clip`, `totalfm`) để dispatch tới builder.
- `__init__.py` — re-export `BaseImageEncoder`, `ImageFeatures`, `build_image_encoder`,
  `registered_backbones`.

**`components/experts/`** — `organ_expert.py`: `OrganExpert(BottleneckMLPAdapter)`, chỉ là alias
tương thích ngược cho adapter regional gốc.

**`components/fusion/`** — cách gộp các đặc trưng theo tên (global/heart/PA/lung...):
- `base.py` — `FusionModule(nn.Module, ABC)` (interface dạng mapping) + `ordered_features` (validate
  và sắp thứ tự tensor theo tên).
- `concat_mlp.py` — `ConcatMLPFusion`: nối các đặc trưng có tên (che đặc trưng không khả dụng) rồi
  chiếu qua MLP.
- `concat.py` — `ConcatFusion(ConcatMLPFusion)`: wrapper legacy dùng list có thứ tự.
- `soft_moe.py` — `SoftMoEFusion`: fusion kiểu soft mixture-of-experts có router, trọng số theo đặc
  trưng có tên, có mask khả dụng.
- `base_moe.py` — `DenseSoftMoE(SoftMoEFusion)`: wrapper legacy dùng list có thứ tự.
- `factory.py` — `build_fusion(config, feature_names, feature_dim, ...)`: dựng `ConcatMLPFusion` hoặc
  `SoftMoEFusion` theo config.

**`components/peft/`** — parameter-efficient fine-tuning:
- `freeze.py` — `apply_peft(module, config)`: áp policy `frozen`/`full`/`lora`;
  `trainable_parameter_summary` báo tổng/trainable/LoRA params.
- `lora.py` — `LoRALinear(nn.Module)` + `inject_lora(module, target_modules, ...)`: thay các
  `nn.Linear` khớp bằng bản có LoRA.

**`components/roi/`** — pooling và masking theo vùng:
- `feature_extractor.py` — `ROIFeatureExtractor(nn.Module)`: pool feature map dùng chung thành vector
  theo từng vùng (tim/PA/phổi) qua `mask_guided_pool`.
- `pooling.py` — `resize_mask` (resize mask nhị phân về đúng shape feature map bằng nearest-neighbour)
  và `mask_guided_pool` (mean pooling có trọng số theo mask, xử lý mask rỗng).
- `masks.py` — API masking runtime bằng Torch: `remove_roi`, `keep_only_roi`, `matched_random_mask`,
  `apply_counterfactual` — áp transform xoá/giữ-chỉ ROI (fill hằng số hoặc theo policy) lên volume
  input lúc train/eval, tái dùng `source.roi.counterfactual`/`source.roi.random_controls`.

### 11.3. `source/concepts/` — concept bottleneck (tùy chọn)

- `model.py` — `ConceptBottleneck(nn.Module)`: dự đoán tập concept lâm sàng đã cấu hình tường minh từ
  đặc trưng đã pool, rồi embed dự đoán đó cho fusion downstream.
- `schema.py` — `ConceptSpec` + `enabled_concept_specs(config)`: validate/chuẩn hoá config concept
  bottleneck, bắt buộc mỗi concept phải có nguồn supervision thật (native/validated_silver/
  expert_reviewed).

### 11.4. `source/data/` — I/O và validate dữ liệu

- `dataset.py` — `CTPADataset(Dataset)`: nạp volume/mask/label/EHR/PESI/silver-label từ manifest đã
  validate (kèm tra cứu ROI/silver tùy chọn); `ReportEmbeddingDataset(Dataset)`: dataset chỉ dùng
  report embedding cho baseline `DX04_report_only`; `load_volume` hỗ trợ `.pt/.npy/.npz/.nii(.gz)`.
- `manifests.py` — I/O và validate manifest: `read_rows` (CSV/JSONL/Parquet), `audit_manifest`/
  `ManifestAudit` (chồng lấn split/patient, trùng lặp, thiếu file, kiểm tra số), `audit_report_table`,
  `audit_silver_table`, `patient_ids_for_splits`, `require_valid_manifest`, `create_patient_split`,
  `audit_temporal_holdout`.
- `paths.py` — `ProjectPaths` (dataclass) resolve code/cloud/data/output root từ config + env
  (`discover_code_root`, `code_asset`, `output_asset`, `assert_persistent_output`,
  `require_within_output`) để chặn path escape ra ngoài output root.
- `preflight.py` — `run_preflight`/`require_preflight`/`PreflightReport`/`Check`: cổng validate lớn
  trước khi chạy — kiểm tra path, khả năng ghi/collision output, schema manifest, contract
  model/backbone, yêu cầu supervision theo từng stage (mask/ROI/silver/EHR/PESI), tương thích lineage
  checkpoint (`checkpoint_lineage_errors`), và tính khả dụng compute/CUDA cho mọi stage.
- `cv.py` — cross-validation patient-level tất định: `stratified_patient_kfold_assignments`,
  `repeated_stratified_patient_splits`, `patient_kfold_assignments`, `repeated_kfold_patient_splits`,
  `nested_kfold_patient_splits`, `assign_fold_column` — đảm bảo không patient nào lọt qua nhiều fold.
- `transforms.py` — `Compose`, `CTWindowNormalize`, `ResizeVolume`, `RandomFlip3D`: transform CT
  đơn giản trên tensor.
- `__init__.py` — chỉ có docstring.

### 11.5. `source/distillation/` — knowledge distillation cho ROI-student

- `losses.py` — `knowledge_distillation_loss` (KL/BCE trên soft-target có temperature) và
  `distillation_loss` (kết hợp có trọng số supervised + distillation).
- `teacher.py` — `freeze_teacher`, `FrozenTeacher(nn.Module)`: bọc model để weight ở eval-mode, không
  gradient, không nằm trong optimizer của student.

### 11.6. `source/distributed/` — chạy multi-GPU/DDP

- `launcher.py` — `LaunchSpec` + `build_launch_spec`/`launch`: dựng và chạy subprocess (một process
  hoặc `torchrun`) với `CUDA_VISIBLE_DEVICES` set theo GPU vật lý được chọn.
- `setup.py` — `DistributedContext` + `initialize_distributed`, `wrap_ddp`, `rank_zero_call`: khởi
  tạo/tắt process group, bọc DDP, helper chỉ chạy ở rank 0 rồi broadcast.
- `gather.py` — `gather_objects`, `gather_prediction_rows`: gom/merge prediction theo rank ở rank 0,
  validate trùng lặp/không khớp.
- `__init__.py` — re-export lazy (import trễ để `preflight` CPU-only chạy được mà không cần torch).

### 11.7. `source/engine/` — training/checkpoint/output engine dùng chung mọi stage

- `checkpoint.py` — lineage/schema checkpoint: `REQUIRED_LINEAGE`, `build_checkpoint_lineage`,
  `save_checkpoint_atomic`, `load_checkpoint` (transfer module chọn lọc), `inspect_checkpoint`,
  `checkpoint_sha256` — ép mọi checkpoint lưu ra phải có phần provenance chặt chẽ.
- `experiment.py` — `OutputManager`: resolve thư mục run theo family dưới output root bền vững, xử lý
  resume/overwrite/collision (`prepare`, `inspect_collision`, `ensure_layout`), ghi JSON/bytes atomic,
  `compact_result` (dựng `result.json` chuẩn), `prepare_resumable_run`.
- `factory.py` — `build_task_model(config)`: dựng model diagnosis/prognosis/contour (kèm encoder,
  PEFT, concept bottleneck, biến thể report-only) từ config đã resolve.
- `task_steps.py` — `task_loss_step(config)`: trả về callback loss theo stage (diagnosis/prognosis/
  contour) — áp counterfactual lên input, tính masked multitask loss, auxiliary organ loss, silver
  loss, và distillation tùy chọn với teacher đóng băng.
- `trainer.py` — `Trainer`: vòng lặp train/validate tổng quát theo task — AMP, gradient accumulation,
  reduce loss/metric phân tán, lưu best checkpoint, log lịch sử ra CSV; `move_to_device` chuyển tensor
  trong batch đệ quy.
- `transfer.py` — `transfer_modules(model, checkpoint, modules)`: nạp weight của các sub-module được
  chọn từ checkpoint nguồn vào model đích.
- `__init__.py` — chỉ có docstring.

### 11.8. `source/imaging/` — tiện ích ảnh NIfTI/mask

- `nifti.py` — `combine_masks`, `validate_binary_mask`, `load_nifti`, `save_binary_mask`,
  `geometry_metadata`, `same_geometry`, `binary_dice`, `mask_qc` (QC connected-component/volume).
- `preview.py` — `write_overlay_preview`: render PNG overlay mask lên lát cắt lớn nhất của volume để
  review QC.

### 11.9. `source/metrics/` — đo lường/thống kê

- `bootstrap.py` — `patient_bootstrap` (bootstrap CI patient-level tổng quát), `bootstrap_binary_predictions`, `bootstrap_prognosis_predictions`.
- `calibration.py` — `calibration_slope_intercept` (hồi quy logistic) và `calibration_curve_points`
  (calibration curve dạng bin qua scikit-learn).
- `classification.py` — `select_threshold_on_validation` (Youden's J hoặc F1-optimal) và
  `binary_classification_metrics` (AUROC/AUPRC/accuracy/sensitivity/specificity/PPV/NPV/F1/Brier).
- `paired.py` — `align_predictions_by_patient`, `paired_patient_bootstrap`: bootstrap ghép cặp so
  sánh hai bộ prediction (vd ROI/counterfactual vs reference) trên cùng tập patient held-out.
- `prognosis.py` — `prognosis_metrics`: gộp metric phân loại + slope/intercept calibration.
- `reporting.py` — `stard_ai_checklist`, `tripod_ai_checklist`: dựng khung báo cáo theo hướng
  STARD+AI/TRIPOD+AI (metadata, không phải thuật toán compliance đầy đủ).
- `segmentation.py` — `dice_score`, `surface_metrics` (NSD/HD95 qua SciPy distance transform),
  `segmentation_case_metrics`.
- `silver.py` — `silver_validation_metrics`: AUROC/AUPRC/F1 (binary) hoặc macro-F1/balanced-accuracy
  (multiclass) cho từng silver target.
- `__init__.py` — re-export `patient_bootstrap`, `binary_classification_metrics`,
  `select_threshold_on_validation`.

### 11.10. `source/pretraining/` — các stage pretraining

- `foundation.py` — `FoundationModel(nn.Module)`: namespace checkpoint mỏng bọc image encoder cho
  stage "foundation materialization" (F01–F03, không train).
- `silver_supervised.py` — `SilverEncoderAdaptationModel` (encoder C0 + head tạm theo target),
  `silver_adaptation_loss` (masked multitask loss chuẩn hoá theo study), `summarize_silver_labels`
  (thống kê coverage train/validation). Dùng cho SE01.

**`pretraining/alignment/`** — `image_report.py`: `ImageReportAlignment(nn.Module)` — model đối chiếu
ảnh-report kiểu InfoNCE đối xứng, tạo ra **C0** (AL01).

**`pretraining/dapt/`** — các objective DAPT:
- `base.py` — `DAPTObjective(nn.Module)` (lớp cơ sở) + `build_dapt(name, encoder, config)` (factory
  dispatch none/MAE/DINO/SimCLR/anatomy-aware, tương ứng D00–D04).
- `none.py` — `NoDAPT` (D00): no-op, chỉ trả global embedding.
- `mae.py` — `MaskedAutoencoderDAPT` (D01): masking patch ngẫu nhiên + decoder Conv3d 1x1 tái tạo.
- `dino.py` — `DINODAPT` (D02): self-distillation student/teacher (EMA momentum), cross-entropy softmax.
- `simclr.py` — `SimCLRDAPT` (D03): contrastive hai view kiểu NT-Xent.
- `anatomy_dapt.py` — `AnatomyAwareDAPT` (D04): đối chiếu embedding global vs anatomy-ROI-only
  (`keep_only_roi`).

### 11.11. `source/profiling/`

- `model_profile.py` — `profile_model`: đo latency, VRAM đỉnh, GFLOPs (qua `torch.profiler`), và số
  tham số trainable của một model/forward callable.

### 11.12. `source/roi/` — xây dựng ROI offline (khác `components/roi/` là runtime pooling/masking)

- `builder.py` — pipeline xây ROI offline: `build_roi_dataset`/`_process_study`/
  `compact_roi_manifest` — dựng và QC 8 mask ROI định nghĩa sẵn (ROI1–ROI8, gồm cả matched random
  control) theo từng study từ output segmentation, có cache, preview, chạy song song.
- `counterfactual.py` — `MaskingPolicy` (dataclass: chiến lược thay thế zero/global_mean/local_mean/
  noise_matched) + `apply_mask_transform`: transform giữ/xoá mask không phụ thuộc NumPy/Torch, dùng
  chung cho cả build ROI offline lẫn counterfactual runtime.
- `masks.py` — tiện ích hình học mask bằng NumPy: `union_masks`, `subtract_masks`, `physical_ball`,
  `dilate_mask`, `body_mask_from_hu`.
- `random_controls.py` — `stable_control_seed` (seed tất định theo patient/study/control) và
  `matched_random_control`: dựng mask control ngẫu nhiên đúng thể tích, không chồng lấn, nằm trong
  body/body-wall cho một ROI mục tiêu.

### 11.13. `source/segmentation/` — pseudo-anatomy (SEG01)

- `totalsegmentator.py` — `TotalSegmentatorRunner`/`TASK_CLASSES`: wrapper subprocess gọi CLI
  TotalSegmentator, sinh mask theo từng vùng giải phẫu chính (phổi, tim, trung thất, PA, mạch máu,
  body, và các mask dẫn xuất body-wall/hilar-vessels/strict-heart) kèm ghi provenance đầy đủ.
- `lungmask.py` — `LungMaskRunner`: wrapper subprocess gọi CLI `lungmask` với checkpoint pin cứng,
  verify geometry output.
- `pipeline.py` — `generate_pseudo_anatomy`/`_process_study`: điều phối TotalSegmentator + LungMask
  theo từng study trên nhiều GPU worker, chạy QC (`mask_qc`, Dice phổi cross-model), ghi preview, có
  cache/resumable.
- `__init__.py` — chỉ docstring; cố ý không import ngầm các wrapper cụ thể.

### 11.14. `source/silver/` — sinh nhãn silver từ report text

- `schema.py` — `TargetSpec`/`TARGET_SPECS`/`TARGETS` (schema chuẩn 18 target cho báo cáo PE),
  `SilverLabel` (dataclass validate chặt), `canonical_target`, `validate_target_value`,
  `decode_storage_value`, `validate_label_rows`, `wide_training_rows`.
- `rules.py` — `apply_rule`/`apply_rules`: trích xuất theo regex tất định (`RULE_VERSION
  pe_rules_v2`) cho các target PE presence, acuity, vị trí, dấu hiệu RV/LV, tràn dịch, bệnh phổi mạn,
  có nhận diện ngữ cảnh/uncertainty.
- `providers.py` — `ProviderIdentity`, `StructuredProvider` (protocol), `TransformersProvider`
  (wrapper inference LLM local qua HF causal-LM), `parse_json_response` (validate schema/evidence
  offset chặt của output JSON model).
- `falcon.py` — `FalconExtractor`: bọc `StructuredProvider`, validate model ID, parse/validate JSON
  trích xuất có cấu trúc.
- `medgemma.py` — `MedGemmaExtractor(FalconExtractor)`: cùng contract, model y tế riêng đã pin.
- `adjudicator.py` — `adjudicate(...)`: phân xử theo đồng thuận giữa Falcon và MedGemma (SL02) —
  accept nếu đồng thuận, abstain nếu không.
- `confidence.py` — `route_confidence(response, threshold)`: cổng ngưỡng confidence đơn giản.
- `generator.py` — `SilverGenerator`: điều phối sinh nhãn SL00 (chỉ MedGemma)/SL01 (rule+Falcon)/SL02
  (rule+Falcon+MedGemma adjudicated) theo từng report/target, tạo `SilverLabel` + `AuditRecord`.
- `audit.py` — `AuditRecord` (dataclass) + `evidence_from_outputs`: lưu vết audit đầy đủ cho từng
  quyết định (tách riêng khỏi bảng train gọn), trích đoạn evidence đầu tiên có sẵn.
- `qc.py` — `silver_qc_summary`: dựng artifact QC cho silver generation (tỷ lệ hiện diện, tỷ lệ
  accept/abstain, mức bất đồng giữa provider, kiểm tra giá trị bất khả thi, missingness, sample
  preview).
- `__init__.py` — re-export `TARGETS`, `SilverGenerator`, `validate_target_value`.

### 11.15. `source/tasks/` — model cho từng bài toán lâm sàng

**`tasks/diagnosis/`**
- `model.py` — `DiagnosisModel(nn.Module)` + `DEFAULT_TARGETS`: image encoder → trích đặc trưng
  ROI/organ-adapter → fusion (soft-MoE/concat) → head chẩn đoán chính + auxiliary head theo từng
  organ (tùy chọn).
- `heads.py` — `DiagnosisHeads(nn.Module)`: head linear riêng cho từng target theo
  `target_output_dims`.
- `losses.py` — `diagnosis_loss` (masked multitask BCE/CE trên nhiều target) và
  `organ_auxiliary_loss` (loss auxiliary theo từng organ/target, tra cứu nhãn theo nguồn).
- `organ_targets.py` — `OrganTarget` (dataclass), `DEFAULT_ORGAN_TARGET_MAPPING`,
  `normalize_organ_target_mapping`, `validate_organ_target_supervision`: validate mỗi auxiliary organ
  target phải có supervision thật (native/silver/expert-reviewed).
- `report_baseline.py` — `ReportOnlyDiagnosisModel(nn.Module)`: baseline chỉ dùng NLP
  (`DX04_report_only`), tiêu thụ report embedding có sẵn, **không bao giờ nhìn thấy volume CTPA**.

**`tasks/prognosis/`**
- `model.py` — `PrognosisModel(nn.Module)`: model fusion đa phương thức (ảnh/EHR/PESI, mỗi cái tùy
  chọn) với organ adapter riêng theo từng modality và fusion cấu hình được.
- `concept_model.py` — `ConceptBottleneckPrognosisModel(nn.Module)`: pipeline tùy chọn ảnh → đặc
  trưng organ → concept bottleneck → (+lâm sàng) → tiên lượng, tách riêng khỏi model multimodal
  chính.
- `heads.py` — `PrognosisHead(nn.Sequential)`: MLP 2 lớp cho ra logit tử vong.
- `losses.py` — `mortality_loss`: BCE có mask cho target tiên lượng chính.

**`tasks/contour/`**
- `model.py` — `ContourModel(nn.Module)`: image encoder + `ContourDecoder` cho bài toán tinh chỉnh
  phân đoạn/contour.
- `decoder.py` — `ContourDecoder(nn.Module)`: decoder Conv3d nhỏ sinh logit phân đoạn theo từng
  vùng, upsample về shape volume.
- `losses.py` — `soft_dice_loss`, `contour_loss` (BCE có trọng số + soft-Dice).

### 11.16. `source/utils/` — tiện ích nền tảng

- `config.py` — `ConfigError`, `deep_merge` (hỗ trợ directive `_replace_`/`_delete_`),
  `expand_environment`, `_load_with_bases` (kế thừa `_base_`), `apply_overrides`, `parse_devices`,
  `infer_compute_strategy`, `validate_config` (validate toàn bộ experiment config: stage/id-prefix
  contract, run_scope, compute strategy, config hash), `load_config`, `dump_config`.
- `console.py` — `experiment_header`, `final_evaluation_block`, `silver_block`: in tóm tắt human-
  readable lúc bắt đầu/kết thúc experiment.
- `environment.py` — `git_state`, `environment_report`: ghi lại commit/branch/dirty git cùng thông
  tin Python/PyTorch/CUDA/GPU phục vụ reproducibility.
- `logger.py` — `RunLogger`: ghi dòng log có timestamp vào file log của run, tùy chọn echo ra console.
- `seed.py` — `seed_everything(seed, deterministic)`: seed Python/NumPy/PyTorch (và cờ
  determinism CUDA/cuDNN).

---

## 12. Ý nghĩa `tools/` (CLI — entrypoint thật)

`tools/` là lớp CLI: các entrypoint `argparse` mỏng (mỗi file có `main()`, gọi được qua `python` hoặc
qua `tools/launch.py`) — đọc config YAML qua `source.utils.config`, resolve `ProjectPaths`, chạy cổng
`source.data.preflight`, ghép các khối trong `source/` lại (dựng dataset, dựng model qua
`source.engine.factory`, chạy `source.engine.trainer.Trainer`), rồi lưu kết quả/checkpoint/lineage qua
`source.engine.experiment.OutputManager`.

| File | Vai trò |
|---|---|
| `_common.py` | Hạ tầng CLI dùng chung: `base_parser`, `resolve_cli_config`, `resolve_manifest`, `select_patient_rows`, `PrecomputedReportDataset`, `build_dataset` (dựng `CTPADataset`/`ReportEmbeddingDataset` từ config), `build_training_lineage` (dựng lineage đầy đủ, gồm cả provenance patient-ID), `write_parquet_atomic`, `write_csv_atomic`, `import_symbol`. |
| `preflight.py` | Chạy `run_preflight` và in báo cáo JSON; exit code phản ánh pass/fail. |
| `launch.py` | **Entrypoint chính**: resolve config, chạy preflight, rồi launch đúng entrypoint theo `experiment.stage` (bảng `ENTRYPOINTS`) dưới dạng subprocess trên CPU/1 GPU/DDP; xử lý cờ chọn patient/case cho silver và counterfactual. |
| `launch_parallel.py` | `Job` (dataclass) đọc mục `parallel:`, chia GPU thành group, dựng kế hoạch chạy nhiều `tools/launch.py` theo wave, preflight tất cả trước, chạy tuần tự từng wave và fail-fast giữa các wave. |
| `build_summary.py` | Quét toàn bộ `result.json` dưới output root, dựng bảng CSV tóm tắt theo family và tổng hợp để phục vụ báo cáo. |
| `sync_project.py` | Copy source/configs/scripts (so sánh theo hash) từ repo local sang cloud project root, không xoá gì ở đích (non-destructive). |
| `data/create_split.py` | Wrapper CLI cho `source.data.manifests.create_patient_split` — tạo tường minh một split train/validation/test patient-level tất định. |
| `create_masks/generate_masks.py` | Resolve config/paths/GPU segmentation, chọn patient/case từ manifest, gọi `source.segmentation.pipeline.generate_pseudo_anatomy`, ghi manifest segmentation + `result.json`. |
| `build_rois/build_rois.py` | Đọc manifest của một run segmentation đã hoàn tất, chọn patient/case, gọi `source.roi.build_roi_dataset` để dựng/QC ROI1–8, ghi manifest ROI + `result.json`. |
| `silver_labels/generate_silver_labels.py` | Chạy `SilverGenerator` (SL00/SL01/SL02) với provider Falcon/MedGemma đã cấu hình trên các shard report (sharding phân tán), cache theo report, gather/merge, ghi `labels.parquet`, `audit.jsonl`, QC summary. |
| `pretrain_model/materialize_foundation.py` | Dựng/profile một backbone ảnh public đã pin (`FoundationModel`), lưu làm checkpoint khởi đầu (**không train**) kèm lineage/profile đầy đủ. |
| `pretrain_model/train_dapt.py` | Train DAPT in-domain (none/MAE/DINO/SimCLR/anatomy-DAPT) qua `Trainer`, hỗ trợ DDP, PEFT, ghi lineage/result. |
| `pretrain_model/train_alignment.py` | Train `ImageReportAlignment` (đối chiếu ảnh-report tạo **C0**) dùng report embedding tính sẵn qua `PrecomputedReportDataset`, tùy chọn transfer module từ checkpoint nguồn. |
| `pretrain_model/train_silver_encoder.py` | Adapt encoder C0 dùng silver label đã accept (`SilverEncoderAdaptationModel`) → **C_silver**, gồm validate target contract, trọng số loss theo target, tóm tắt coverage silver label, gather validation phân tán, metric theo từng target. |
| `tasks/train_task.py` | Dựng và train end-to-end model diagnosis/prognosis/contour qua `Trainer`/`task_loss_step`, fit clinical preprocessor trên train, tùy chọn tính distillation validation row, profile model, ghi result/lineage cuối. |
| `tasks/evaluate.py` | Nạp checkpoint đã train, đánh giá diagnosis/prognosis (chọn threshold trên validation, bootstrap metric trên test, calibration curve tùy chọn, so sánh paired với reference) hoặc contour (Dice/NSD/HD95 bootstrap theo patient/vùng); ghi predictions/result/reporting checklist; hỗ trợ smoke-scope (`--patient-id`/`--max-cases`). |
| `tasks/counterfactual.py` | Inference counterfactual "remove-ROI" trên checkpoint đóng băng — tính prediction gốc vs counterfactual (xoá vùng hoặc xoá bằng matched-random), threshold lấy từ validation, paired bootstrap deltas, ghi predictions/result (**không train lại, không chạy lại segmentation**). |

Mỗi thư mục con (`build_rois/`, `create_masks/`, `data/`, `pretrain_model/`, `silver_labels/`,
`tasks/`) có `__init__.py` chỉ mang docstring, không export logic.

---

## 13. Ý nghĩa `scripts/` (Bash wrapper)

Scripts chỉ là wrapper mỏng quanh `tools/*.py`; **logic khoa học không nằm trong shell script**. Chạy
từ Git Bash/WSL/Linux; trên PowerShell thuần nên gọi thẳng lệnh Python tương ứng.

| Folder | Chức năng |
|---|---|
| `foundation/` | F01/F02/F03 materialization (`F01_ct_fm.sh`, `F02_ct_clip.sh`, `F03_totalfm.sh`) |
| `dapt/` | D00–D04 (`D00_none.sh` … `D04_anatomy.sh`) |
| `alignment/` | `AL01_image_report.sh`, tạo C0 |
| `silver_encoder/` | `SE01_train.sh`, tạo C_silver |
| `diagnosis/`, `prognosis/`, `contour/` | train rồi tự evaluate downstream task (`DX01_train.sh`, `PR01_train.sh`, `CT01_train.sh`) |
| `segmentation/` | `SEG01_generate_masks.sh` |
| `roi/` | `ROI01_build.sh` |
| `silver/` | `SL00_medgemma_only.sh`, `SL01_rules_falcon.sh`, `SL02_hybrid.sh` |
| `ablations/` | wrapper ablation runnable (`A01_global_only.sh`, `RM01_remove_pa.sh`, `RS01_pa_only.sh`, `SA01_sl00.sh`, `TR01_public_to_contour.sh`) |
| `parallel/` | `run_batch.sh` — batch launcher chung cho mọi stage hỗ trợ |
| `setup/` | `sync_project_to_cloud.sh` — sync source utility |
| `_common.sh` | helper nội bộ (không gọi trực tiếp): `run_training_experiment` gọi `tools/launch.py`, rồi (với diagnosis/prognosis/contour) gọi `tools/tasks/evaluate.py`, rồi `tools/build_summary.py` |

GPU list là positional argument đầu tiên hoặc biến `GPUS`:

```bash
bash scripts/dapt/D02_dino.sh 0,1
GPUS=0,1 bash scripts/diagnosis/DX01_train.sh
```

Sample selection cho segmentation/ROI/silver dùng biến môi trường `PATIENT_ID`, `MAX_CASES`/
`MAX_REPORTS`, `ALLOW_FULL=1` (không đặt đồng thời `PATIENT_ID` và `ALLOW_FULL`). Các config runnable
không bắt buộc phải có wrapper riêng — luôn có thể gọi thẳng `tools/launch.py --config ...`.

---

## 14. Ý nghĩa `third_party/`

```text
third_party/
├── repos/         clone sạch của upstream source (KHÔNG sửa để nhét logic riêng của project)
├── weights/       model artifact local (bị .gitignore, không commit)
├── versions.yaml  URL, commit ghim, weight kỳ vọng, checksum, trạng thái — nguồn sự thật cần commit
└── README.md
```

Adapter tích hợp phải nằm trong `source/components/` hoặc module project tương ứng, không sửa trực
tiếp code upstream.

### 14.1. `third_party/repos/` — các repo đã clone

| Repo | Vai trò trong pipeline |
|---|---|
| `TotalSegmentator/` | Nguồn pseudo-anatomy **canonical** (SEG01) — segment toàn thân đa cơ quan. |
| `lungmask/` | Model QC **độc lập** cho phổi — Dice thấp có thể flag case, nhưng **không thay** mask phổi của TotalSegmentator. |
| `CT-CLIP/` | Ứng viên foundation backbone (đối chiếu ảnh-text CT, kèm `CT_CLIP/`, `transformer_maskgit/`, `text_classifier/`). |
| `CT-FM/` | Ứng viên foundation backbone (self-supervised 3D CT, kèm nhiều framework SimCLR/SimSiam/VICReg/SwAV, backbone MedNeXt/ResNet/UNet3D). |
| `TotalFM/` | Ứng viên foundation backbone dạng organ-patch (ViT-3D + ModernBERT). |

Không đổi `status` trong `versions.yaml` thành `ready` trước khi kiểm tra bằng checkpoint thật:
checksum/model/revision chính xác, factory import path, quy ước tiền xử lý input, output adapter và
feature dimension, khả năng strict-load theo config.

### 14.2. `third_party/weights/` — layout weight local kỳ vọng

```text
third_party/weights/
  segmentation/
    totalsegmentator/   toàn bộ offline TotalSegmentator task weights
    lungmask/R231.pth   checkpoint LungMask QC chính xác
  foundation/
    ct_fm/model.ckpt
    ct_clip/model.pt
    totalfm/totalfm_en_checkpoint_best_loss.pt
  silver/
    falcon/             thư mục Hugging Face model đầy đủ
    medgemma/            thư mục Hugging Face model đầy đủ
```

Silver generation mặc định `local_files_only: true` (không âm thầm tải model khác). Sau khi thêm/thay
weight: tính SHA-256 → ghi vào `versions.yaml` → điền adapter contract nếu là backbone → chạy lại
preflight cho experiment dùng weight đó.

### 14.3. `third_party/versions.yaml`

Ghi mỗi component: `repo_url`, `commit` (ghim cứng), `expected_weight(_directory)`,
`checksum_sha256`, `status`. Hiện trạng (xem mục 8): `ct_fm`/`ct_clip`/`totalfm` ở trạng thái
`repo_ready_weight_missing_adapter_pending`; `totalsegmentator`/`lungmask` ở
`repo_ready_weights_missing_install_required`; `falcon`/`medgemma` ở `weight_missing` — tất cả
`checksum_sha256: null` cho tới khi có weight thật.

---

## 15. `docs/` và `paper/`

- `docs/anatomy_aware_gap_analysis.md` — audit thực hiện **trước** khi cài đặt phần anatomy-aware:
  liệt kê module có thể tái dùng (`components/roi/pooling.py`, `components/adapters/organ.py`,
  `components/fusion/`, `source/silver/`, `source/roi/counterfactual.py`,
  `source/roi/random_controls.py`, `source/distillation/`, `source/data/manifests.py`/`cv.py`,
  `source/metrics/`, `source/engine/`...), 12 gap đã phát hiện (thiếu interface z_global/z_heart/
  z_pa/z_lung thống nhất, thiếu per-organ loss decomposition, alias silver không khớp tên canonical,
  thiếu arm random-init full-CT, remove-ROI chưa có paired probability delta, thiếu `tests/`...), và
  danh sách external contract chưa giải quyết (checkpoint factory/feature_dim CT-FM/CT-CLIP/TotalFM,
  manifest INSPECT/RSNA thật, cột report embedding, 32 biến EHR, ngày acquisition, expert segmentation
  annotation...).
- `docs/anatomy_aware_implementation.md` — bản đồ triển khai sau khi implement: danh sách toàn bộ
  experiment ID mới đã thêm (DX14–DX22, CF01–CF04, RS06–RS09, KD01–KD04, PR18–PR27,
  SEG02_CTPA_FINETUNE), bảng ánh xạ requirement → file implement → trạng thái, danh sách lệnh mẫu, và
  "Remaining prerequisites" (điền contract backbone thật, tạo manifest/SEG01/ROI01 thật, tạo bảng
  silver accepted thật, train `DX18_ANATOMY_FULL` trước khi chạy CF/KD, bổ sung expert segmentation
  annotation cho `SEG02_CTPA_FINETUNE`).
- `paper/silver label.pdf` — tài liệu tham khảo liên quan tới phương pháp silver-label (không được
  đọc chi tiết trong lần tổng hợp này; mở trực tiếp nếu cần nội dung).

---

## 16. Tệp cấu hình dự án (root)

- `pyproject.toml` — package `pe-project` 0.1.0, build bằng `setuptools`; khai báo dependency core +
  3 nhóm optional (`training`, `segmentation`, `dev`); `packages.find.include = ["source*", "tools*"]`
  (tức là chỉ `source/` và `tools/` được đóng gói, không đóng gói `third_party/`); cấu hình `pytest`
  trỏ `testpaths = ["tests"]` (thư mục hiện chưa tồn tại — xem mục 1) và `ruff` (line-length 110,
  target py311).
- `requirements.txt` — pin version cụ thể (khoảng `<major-tiếp-theo>`) cho toàn bộ dependency, có
  comment giải thích từng nhóm (core, 3D training/I/O ảnh y tế, TotalSegmentator/LungMask pin,
  foundation-model/silver-model adapter, dev).
- `.gitignore` — bỏ qua cache/venv/pycache; **không** commit `cache/*`, `outputs/`, mọi checkpoint
  (`*.ckpt/.pt/.pth/.safetensors`), ảnh (`*.nii(.gz)`), bảng dữ liệu (`*.parquet`), và nội dung
  `third_party/repos/*`/`third_party/weights/*` (trừ `.gitkeep`/README) — đúng như mô tả "output khoa
  học sống trên cloud-backed storage, không nằm trong Git".

---

## 17. Tóm tắt kiến trúc (đọc nhanh)

`source/` là thư viện thuần: I/O dữ liệu (`data/`), encoder ảnh/lâm sàng + fusion/adapter/PEFT
(`components/`), model theo từng task (`tasks/`), engine train/checkpoint/output dùng chung
(`engine/`), và các helper cross-cutting (`metrics/`, `imaging/`, `utils/`, `distributed/`). `tools/`
là lớp CLI — entrypoint mỏng đọc config YAML, resolve path, chạy `preflight`, ghép các khối thư viện
lại, rồi lưu kết quả/checkpoint/lineage. `tools/launch.py` và `tools/launch_parallel.py` là entrypoint
thật được `scripts/*.sh` gọi, dispatch tiếp sang các tool theo stage (`pretrain_model/*`,
`silver_labels/*`, `tasks/*`, `build_rois/*`, `create_masks/*`).

Pipeline khoa học đi theo chuỗi giai đoạn, mỗi giai đoạn khớp với một subpackage: `segmentation/` sinh
pseudo-anatomy mask (TotalSegmentator + LungMask) kèm QC; `roi/` suy ra và QC 8 mask ROI/counterfactual-
control (ROI1–ROI8) từ các mask đó; `silver/` khai thác nhãn phụ trợ từ report PE bằng rule + LLM
provider (Falcon/MedGemma) kèm adjudication, confidence routing, audit trail; `pretraining/` phủ từ
materialize checkpoint nền tảng, self-supervision DAPT (MAE/DINO/SimCLR/anatomy-aware), đối chiếu
ảnh-report, tới silver-supervised encoder adaptation — sinh chuỗi checkpoint kế tiếp nhau (public →
C0 → C_silver); `tasks/` (diagnosis, prognosis, contour) dựng model lâm sàng downstream tiêu thụ các
encoder đó cộng thêm organ-adapter/fusion/concept-bottleneck; `engine/` là training loop dùng chung,
contract lineage checkpoint, và quản lý output/run directory cho mọi stage; `distillation/` cung cấp
loss knowledge-distillation từ teacher đóng băng khi train model "student" chỉ dùng ROI, được đánh giá
sufficiency/counterfactual qua `tools/tasks/evaluate.py` và `counterfactual.py`. Lineage checkpoint
(`source/engine/checkpoint.py`) và validate `preflight` (`source/data/preflight.py`) là hai cơ chế
đảm bảo tính chặt chẽ khoa học xuyên suốt toàn bộ chuỗi này (không leakage ngầm, không giá trị mặc
định ngầm).
