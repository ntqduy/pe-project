# Ý tưởng toàn bộ bài nghiên cứu

Cập nhật theo repository ngày 2026-09-13. Repo:
`E:\PE_NU\source\pe-project`.

Tài liệu này là bản mô tả thống nhất cho câu hỏi khoa học, thiết kế thí nghiệm và cách code
thực thi. Nó không tuyên bố kết quả khi thí nghiệm chưa chạy. Nguồn sự thật về experiment là
`configs/experiments.yaml`; nguồn sự thật về trạng thái artifact là `run.py plan/preflight`.

## 1. Ý tưởng trung tâm

Tên làm việc:

> An anatomy-partitioned multimodal foundation model with counterfactual erasure for
> pulmonary embolism diagnosis and prognosis on 3D CTPA.

Câu hỏi chính: một encoder CTPA 3D được thích nghi với domain, căn chỉnh với report và tiếp
tục học từ silver labels có tạo representation tốt hơn cho chẩn đoán PE và tiên lượng hay
không? Nếu metric tăng, cải thiện đến từ thông tin giải phẫu liên quan PE hay chỉ từ vùng ảnh
ngẫu nhiên có cùng kích thước? Mô hình đã khóa có tổng quát sang Turkey external test hay không?

Bài không chỉ đề xuất một classifier. Đóng góp dự kiến là một chuỗi bằng chứng:

1. so sánh công bằng public backbone, DAPT, image-report C0 và silver-adapted C_silver;
2. diagnosis single-task và native multitask trên cùng cohort/split;
3. giải thích anatomy bằng necessity, sufficiency và matched random control;
4. chuyển encoder sang prognosis đa modality và nhiều horizon;
5. external validation test-only chính trên Turkey; RSPECT là nhánh supervised transfer;
6. toàn bộ mask, label, checkpoint và prediction có lineage/QC để tái lập.

## 2. Các giả thuyết

- H1: DAPT trên CTPA cải thiện representation so với published/original backbone.
- H2: image-report alignment cải thiện hơn DAPT đơn thuần và tạo checkpoint C0.
- H3: silver supervision có abstention tạo C_silver tốt hơn C0 cho diagnosis; lợi ích phải
  được báo cùng coverage, không chỉ precision trên tập nhỏ đã chọn.
- H4: native multitask (`pe_positive`, `pe_acute`, `pe_subsegmental`) giúp primary PE task
  hơn single-task khi giữ nguyên mọi yếu tố khác.
- H5: PA/heart/lung mang thông tin khác nhau; thay đổi thật phải lớn hơn control ROI8 cùng
  hình dạng và thể tích.
- H6: C0/C_silver/C_diagnosis cung cấp initialization hữu ích cho prognosis hơn backbone gốc.
- H7: improvement nội bộ còn giữ được trên Turkey test-only; RSPECT evaluation chỉ bổ sung.

Đây là giả thuyết cần kiểm định, không phải kết luận trước dữ liệu.

## 3. Dataset và đơn vị phân tích

### INSPECT

INSPECT là development dataset cho preprocessing, representation learning, diagnosis và
prognosis. Split official được giữ theo patient. Một patient không được xuất hiện ở nhiều
split. Raw CTPA/report/EHR là read-only; mọi artifact dẫn xuất ghi sang derived/output root.

Các manifest chính:

- `ctpa.csv`: mọi study đủ điều kiện;
- `diagnosis.csv`: study và ba native labels;
- `paired_reports.csv`, `reports.csv`: alignment và silver extraction;
- `prognosis.csv`: legacy confirmed acute-PE/30-day contract;
- `prognosis_all_patient.csv`: cohort canonical `all_comers`;
- `prognosis_pe_positive.csv`: cohort canonical `pe_positive_only`;
- `prognosis_cohort_membership.csv`: included/excluded và lý do cho từng patient/cohort.

### RSPECT và Turkey

`RSPECT` là RSNA-STR Pulmonary Embolism Detection. Theo email, đây là nguồn public hữu ích cho
pretraining/supervised transfer, không thay vai trò của Turkey. Turkey (email ước tính khoảng
700 scan nhưng chưa xác nhận) là external test chính trong proposal. Repo giữ RSPECT ở nhánh
`repr.rspect.*` và có `diag.external.turkey_test` riêng. Cả hai hiện blocked vì chưa có normalized
manifest/volume phù hợp; không hard-code số lượng chưa được kiểm chứng.

## 4. Pipeline khoa học

```text
Published 3D CT backbone
        |
        +--> fixed frozen probe ----------------------- baseline representation
        |
        +--> DAPT trên INSPECT CTPA ------------------ C_dapt
                  |
                  +--> image-report alignment -------- C0
                          |\
                          | +--> diagnosis từ C0
                          | +--> prognosis từ C0
                          | +--> RSPECT supervised-transfer side branch
                          |
                          +--> accepted silver training -> C_silver
                                      |\
                                      | +--> diagnosis từ C_silver
                                      | +--> prognosis từ C_silver
                                      +--> representation probe

Diagnosis checkpoint C_diagnosis ---------------------> prognosis initialization

Segmentation masks -> semantic ROI + ROI8 controls ---> anatomy branches/ablation/QC
Frozen selected diagnosis model ----------------------> Turkey test-only evaluation
```

Dataset build, segmentation và silver extraction có thể chạy độc lập. ROI phụ thuộc mask
segmentation của đúng study nhưng nhiều study có thể chạy CPU song song. Training chỉ bắt đầu
khi manifest, split audit, checkpoint lineage và input artifact qua preflight.

## 5. Preprocessing và chống leakage

CT được canonicalize orientation/spacing, crop theo body và fit vào tensor target shape theo
config. Mỗi cache có geometry/preprocessing fingerprint. Cùng preprocessing contract phải
được dùng cho mọi arm được so sánh.

Quy tắc leakage:

- giữ official patient split; kiểm tra patient overlap trước và sau preprocessing;
- prognosis lấy một index study cho mỗi patient/cohort;
- EHR chỉ lấy event trong window được cấu hình quanh index time;
- imputation/normalization chỉ fit trên train;
- class weights/sampling nếu dùng chỉ tính từ train;
- threshold và model selection chỉ dùng validation;
- test không dùng để tune, calibrate hay fine-tune;
- external test-only config bị `train_task.py` từ chối.

## 6. Segmentation, ROI và anatomy hypothesis

TotalSegmentator tạo pseudo-anatomy masks; LungMask chỉ QC chéo lung, không âm thầm thay thế
mask. Mỗi anatomy lưu NIfTI canonical, tối đa ba PNG review và row QC.

ROI mapping sau audit:

| ROI | Semantic name | Định nghĩa | Dùng cho |
|---|---|---|---|
| ROI1 | `heart_mediastinum` | heart union mediastinum | keep-only |
| ROI2 | `strict_heart` | heart nghiêm ngặt | keep-only |
| ROI3 | `dilated_central_pulmonary_artery` | central PA dilation | removal |
| ROI4 | `pulmonary_artery_tree` | PA tree | keep-only |
| ROI5 | `whole_lung_with_vessels` | toàn phổi | keep-only |
| ROI6 | `lung_parenchyma_without_large_vessels` | lung trừ vessel lớn | keep-only |
| ROI7 | `heart_hilar_vessels_exclusion` | heart union hilar vessels | removal |
| ROI8 | `control_for_<source>` | rigid translation ROI2/4/6 | matched control |

Không tạo `ROI1_full_ct`: full CTPA đã là baseline input. Không đổi số ROI vì sẽ thay nghĩa
experiment lịch sử.

ROI8 giữ nguyên mask source bằng rigid translation, cùng voxel count và physical volume,
nằm trong body, giữ z-range mặc định và không giao source hoặc union anatomy cấm. Seed được
tạo từ patient, study, source ROI và config seed. Không đủ chỗ sau `max_attempts` thì
`failed/no_valid_control_location`; không co mask và không chọn một đám voxel gần anchor.

Anatomy analysis gồm:

- necessity: cùng frozen diagnosis model, xóa heart/PA/lung/control rồi tính paired delta;
- sufficiency: train ROI-only student từ cùng initialization, với và không knowledge
  distillation;
- sufficiency cho prognosis: bốn arm `prog.{heart,pa,lung,random}_student`
  (`configs/runs/04_prognosis/students/`) — cùng kiến trúc `prog.image`, image-only, không
  KD, đọc theo hàng trần `prog.image` và sàn `prog.random_student`;
- matched control: mọi kết luận anatomy phải vượt thay đổi của ROI8 tương ứng.

## 7. Representation learning

### Published backbone

Đóng gói checkpoint gốc với adapter output được inspect thật. Không được đoán factory,
feature dimension hoặc remap weight. Fixed probe đóng băng encoder và chỉ train một head nhỏ.

### DAPT

Các arm: none, MAE, DINO, SimCLR và anatomy-aware DAPT. Mọi arm giữ dataset, preprocessing,
optimizer và evaluation contract; chỉ objective thay đổi. DAPT train full encoder.

### Image-report alignment

Image encoder được căn chỉnh với precomputed report embeddings bằng symmetric InfoNCE, tạo
C0. Text encoder không được chạy lại trong stage này. Batch cần ít nhất hai pair.

### Silver adaptation

C0 nhận accepted report-derived auxiliary labels để tạo C_silver. Silver label không thay
native diagnosis ground truth. So sánh C0 và C_silver bằng fixed probe và downstream runs có
cùng data/head/hyperparameters.

## 8. Silver labels

Bảy cascade là rule, Falcon, MedGemma và các tổ hợp theo thứ tự. Rule rõ ràng được ưu tiên;
model chỉ accept khi value hợp lệ và confidence đạt threshold. Hai model phải đồng thuận và
cùng vượt agreement threshold. Không chắc chắn là `abstained`; lỗi inference/data là `failed`.

Output chính:

```text
E:\PE_NU\source\pe-project\outputs\silver_label\<method>\
  silver_labels.csv
  silver_label_confidence.csv
  logs/run.log
```

`silver_labels.csv` là normalized long table, một row cho report-target; chỉ `accepted` vào
loss. `silver_label_confidence.csv` giữ decision, threshold, confidence, reason và evidence.
Provider confidence không bị giả thành positive-class probability. Mọi kết quả silver phải
báo coverage = accepted / eligible cùng performance trên validation.

## 9. Diagnosis

Ba native labels chính xác:

- `pe_positive` từ `pe_positive_nlp`;
- `pe_acute` từ `pe_acute`;
- `pe_subsegmental` từ `pe_subsegmentalonly`.

Missing/censored để trống và mask loss. Mỗi binary head dùng BCE-with-logits; multitask tổng
loss theo weight cấu hình. Hiện chưa áp class weight mặc định. Fine-tuning downstream mặc
định LoRA (`attn`, `projection`, rank 8, alpha 16, dropout 0.05); optimizer chỉ nhận
trainable parameters.

Diagnosis matrix tối thiểu:

| ID khái niệm | Mode | Initialization | Mục đích |
|---|---|---|---|
| EXP-DX-01 | PE single-task | published | baseline |
| EXP-DX-02 | PE single-task | C0 | lợi ích image-report alignment |
| EXP-DX-03 | PE single-task | C_silver | lợi ích silver |
| EXP-DX-04 | native multitask | C0 | lợi ích multitask |
| EXP-DX-05 | native multitask | C_silver | silver + multitask |

DAPT initialization là arm representation bổ sung để tách lợi ích domain adaptation khỏi
image-report alignment; nó không thay năm comparison chính ở trên.

Các anatomy-aware diagnosis arms thêm global/heart/PA/lung experts và so concat MLP,
late-logit, Soft-MoE. Encoder chỉ forward full volume một lần; organ features là mask-pooled
views của cùng feature map, không phải nhiều encoder độc lập.

Metric: AUROC, AUPRC, sensitivity, specificity, F1 và patient bootstrap CI. Operating
threshold được chọn trên validation và khóa cho test.

## 10. Prognosis

Hai cohort độc lập là `all_comers` và `pe_positive_only`. Bảy endpoint thật trong release:

- 1-, 6-, 12-month all-cause mortality;
- 1-, 6-, 12-month readmission;
- 12-month pulmonary hypertension.

Không gọi mortality là PE-related khi data không định nghĩa như vậy. Mỗi endpoint lưu event,
observed/censored, time-to-event và source censor flag. Censored/missing không trở thành 0.

Thiết kế factorial:

| Trục | Giá trị |
|---|---|
| cohort | all_comers, pe_positive_only |
| EHR window | EHR_0_h, EHR_24_h |
| endpoint | bảy endpoint trên |
| modality | clinical-only, PESI, image, image+clinical, image+PESI, cả ba |
| image structure | global, anatomy-aware |
| fusion | concat, late-logit, Soft-MoE |
| initialization | published, C0, C_silver, C_diagnosis; DAPT khi protocol chọn |

PESI/sPESI chỉ được dùng sau clinical approval mapping. Mọi modality comparison phải dùng
cùng patient, split và endpoint. Evaluation dùng classification/discrimination,
calibration khi đủ dữ liệu và bootstrap CI; time-to-event fields được giữ để mở rộng survival
analysis mà không bịa censoring.

## 11. RSPECT transfer và Turkey external evaluation

Bốn mode RSPECT có thể phân biệt:

1. frozen evaluation của diagnosis checkpoint có head tương thích; C0 thuần không có PE head
   nên không được gọi là zero-shot classifier;
2. fixed linear probe train head trên external train, encoder frozen;
3. supervised fine-tuning chỉ trên external train, model selection trên validation;
4. external test-only evaluation của một checkpoint đã chọn.

Chuỗi Turkey test-only là normalize -> preprocessing -> segmentation -> ROI -> inference
-> diagnosis metrics. Config `diag.external.turkey_test` bắt buộc truyền `--checkpoint`, đọc
threshold đã khóa từ artifact INSPECT validation và chỉ evaluate Turkey test. Không chọn
threshold trên external cohort và không có thao tác train trong mode này.
Checkpoint phải có architecture/head tương thích; encoder-only C0 cần probe/fine-tuning trên
train trước, không thể strict-load như một classifier hoàn chỉnh.

## 12. Checkpoint flow và PEFT

```text
published -> C_dapt -> C0 -> C_silver
                    \-> C_diagnosis -> C_prognosis
```

Nguồn init được chọn bằng `encoder.init_source`: `pretrained`, `dapt`, `c0`, `silver`,
`diagnosis`, `rspect_single`, `rspect_multitask`, `custom`. Transfer report phải ghi path
tuyệt đối, missing/unexpected keys, shape mismatch, số layer encoder load và head reset.

| Stage | Full FT | Frozen encoder | LoRA | Adapter | Parameter được train |
|---|---:|---:|---:|---:|---|
| DAPT | có | không | không mặc định | objective-specific | encoder + objective |
| alignment | có | không | không mặc định | projection | image projection/encoder |
| fixed probe | không | có | không | probe head | chỉ head |
| diagnosis | hỗ trợ | hỗ trợ | mặc định | organ adapters | LoRA/adapters/heads |
| prognosis | hỗ trợ | hỗ trợ | mặc định | modality/organ adapters | LoRA/adapters/heads |
| evaluation | không | có | không | không | không parameter |

`partial` chưa implement và bị từ chối, không giả là supported. Mỗi `best.ckpt` có sidecar
metadata chứa SHA-256 ID, stage/source, dataset, task/labels, architecture, fine-tuning,
timestamp, git commit và config path.

## 13. Statistics và nguyên tắc so sánh

- Chọn primary endpoint/metric trước khi xem test.
- Report point estimate và patient-level bootstrap CI.
- Với counterfactual, dùng paired bootstrap trên cùng patient.
- Report class prevalence, số observed/censored/missing và số patient mỗi split.
- Với silver, report coverage theo target/source cùng accuracy/precision nếu có validation.
- Không chọn architecture/checkpoint theo external test.
- Nếu chạy nhiều seed, mỗi seed có experiment ID riêng; report mean/dispersion và không ghi
  đè output.
- Mọi so sánh representation giữ fixed probe contract; mọi so sánh fusion giữ encoder,
  cohort, split, optimizer và label giống nhau.

## 14. Output và QC

Segmentation/ROI lưu NIfTI chính, tối đa ba PNG representative slice mỗi anatomy/ROI,
manifest và CSV QC. PNG có CT, overlay bán trong suốt, patient/study/name/z/source/status,
voxel/physical volume và CTPA W/L.

Status chuẩn: `pass`, `failed`, `skipped`, `abstained`. `run.log` ghi UTC timestamp, command,
config, data/split, checkpoint, tiến độ, tổng kết, elapsed time và traceback khi lỗi.

Training output:

```text
E:\PE_NU\source\pe-project\outputs\<family>\<run_id>\
  best.ckpt
  best.ckpt.metadata.json
  resolved_config.yaml
  lineage.json
  environment.json
  metrics.json
  result.json
  logs/run.log
  logs/history.csv
  predictions.parquet          # sau evaluation
```

Không xem `result.json` một mình là đủ để audit; phải giữ config, lineage, checkpoint,
predictions và manifest source.

## 15. Các bảng và hình dự kiến cho bài

- Figure 1: toàn pipeline và checkpoint lineage.
- Figure 2: anatomy-aware shared feature map, organ pooling và fusion.
- Figure 3: semantic ROI/ROI8 overlays minh họa cùng volume nhưng khác vị trí.
- Table 1: cohort flow, split, prevalence, missing/censoring.
- Table 2: fixed-probe comparison published/DAPT/C0/C_silver.
- Table 3: diagnosis single-task/multitask và initialization matrix.
- Table 4: anatomy necessity/sufficiency với ROI8 controls.
- Table 5: prognosis cohort x endpoint x modality/init, kèm organ-only sufficiency
  (heart/PA/lung vs matched random).
- Table 6: Turkey internal-to-external generalization; RSPECT transfer là supplementary.
- Supplement: silver coverage/abstention, QC failure counts, LoRA/trainable parameters,
  checkpoint lineage và hyperparameters.

Chỉ điền kết quả khi artifact và test run tồn tại; không tạo bảng số giả.

## 16. Cách chạy

```bash
# Xem registry và dependency
python run.py list
python run.py plan data.segmentation

# Data support smoke
python run.py run data.segmentation --gpus 0 --max-cases 1
python run.py run data.roi --max-cases 1
python run.py run data.silver.rule --max-reports 10

# Representation/diagnosis sau khi checkpoint thật sẵn sàng
python run.py preflight repr.align
python run.py run repr.align --gpus 0
python run.py run diag.matrix.single_task --gpus 0
python run.py run diag.matrix.multitask --gpus 0

# External test-only; không dùng train_task.py
python tools/tasks/evaluate.py \
  --config configs/runs/03_diagnosis/external/turkey_test.yaml \
  --checkpoint E:\PE_NU\source\pe-project\outputs\diagnosis\<selected_run>\best.ckpt \
  --allow-full
```

Hiểu dữ liệu trước khi train — cohort, prevalence, event count, rule coverage:

```bash
PROFILE=full_inspect bash scripts/ana.sh
```

## Tài liệu liên quan

| File | Nội dung |
|---|---|
| [docs/PROPOSAL_ALIGNMENT.md](docs/PROPOSAL_ALIGNMENT.md) | **Đối chiếu proposal ↔ repo: khớp gì, thiếu gì** |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Bảng kết quả cho paper, kèm lệnh chạy từng bảng |
| [analysis/README.md](analysis/README.md) | EDA dataset profile (`bash scripts/ana.sh`) |

Chi tiết từng stage nằm trong `docs/01`–`docs/08`.

## 17. Blocker và giới hạn hiện tại

- CT backbone factories/output adapters/feature dimensions và real weights chưa được
  xác nhận đầy đủ; image experiments phải fail preflight đến khi inspect xong.
- TotalSegmentator/LungMask và model weights phải được cài/stage để chạy segmentation thật.
- RSPECT normalized data chưa sẵn sàng cho supervised-transfer side branch.
- Turkey path/count/manifest/label schema chưa được giao, nên external test chính còn blocked.
- EHR feature columns và PESI clinical approval còn chặn một số prognosis arms.
- Không có expert embolus contours nên contour stage vẫn deferred.
### Trạng thái gap so với proposal

| # | Hạng mục | Trạng thái | Chạy bằng |
|---|---|---|---|
| G1 | PENet-style baseline | ✅ code/config; ⛔ chặn bởi INSPECT data | `diag.baseline.penet_style` |
| G2 | Turkey external test | ✅ contract; ⛔ chặn bởi Turkey data | `diag.external.turkey_test` |
| G3 | 3-fold patient-level stratified CV | ✅ **đã nối** | `bash scripts/run_cv.sh <config>` |
| G4 | Architecture ablation (6 biến thể) | ✅ **đã thêm** | `bash scripts/run_ablation_arch.sh` |
| G5 | EHR ablation (4 biến thể) | ✅ **đã thêm** | `bash scripts/run_ablation_ehr.sh` |
| G6 | nnU-Net CTPA finetune | ❌ chưa implement; ⛔ thiếu expert masks | — |

**G3** dùng `tools/tasks/train_cv.py`: sinh manifest cho từng fold rồi gọi `train_task.py` và
`evaluate.py` không đổi. Fold chia ở mức **patient**, stratify theo primary target; không
stratify được thì fallback patient K-fold và **ghi lý do** vào `cv_summary.json` chứ không
âm thầm đổi thiết kế.

**G4** giữ nguyên C0 initialization/split/optimizer, chỉ đổi `task.regions` và fusion
component — nên chênh lệch giữa các hàng chỉ đến từ kiến trúc. `full_moe` vs
`full_no_router` cô lập router.

**G5** dùng `task.ehr_include_missingness` vốn đã nối sẵn vào `ClinicalEncoder`. Tắt indicator
làm input width của clinical encoder giảm một nửa, model không còn biết giá trị nào đã impute.
`data.ehr_columns` / `ehr_columns_common` vẫn rỗng — phải điền trước khi chạy thật.

**G6 chưa hoàn thành**: nnU-Net CTPA fine-tune cần expert-reviewed annotation subset mà hiện
chưa có. Không được mô tả nó là “đã freeze segmentation model”; hiện pipeline chỉ có
TotalSegmentator + LungMask cross-check. Đây là blocker và giới hạn phải nêu trong paper.

Chi tiết đối chiếu: [docs/PROPOSAL_ALIGNMENT.md](docs/PROPOSAL_ALIGNMENT.md).
- Unit tests có thể skip NumPy/Torch/NIfTI-dependent cases nếu môi trường thiếu dependency;
  việc skip phải được báo, không gọi là full smoke success.

Các giới hạn này là một phần của báo cáo reproducibility. Không điền giá trị giả, không coi
missing label là negative, không bỏ qua failed ROI và không fine-tune trên test để làm pipeline
trông như đã hoàn thành.
