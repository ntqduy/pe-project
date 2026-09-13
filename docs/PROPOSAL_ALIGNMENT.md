# Audit yêu cầu của giáo ↔ repository

Nguồn chuẩn đã đọc toàn bộ:

- `paper/Gmail - PE study to complete.pdf` — chuỗi email 13 trang, yêu cầu chính ngày
  26/03/2026 và thông tin dữ liệu bổ sung ngày 22/04/2026;
- `paper/Overview_proposal_PE_28_7.pdf` — proposal 17 trang.

Audit code/config/docs ngày 2026-09-13. Ký hiệu:

- **Implemented**: có đường thực thi trong code và contract kiểm tra được;
- **Blocked**: code/config đã có nhưng chưa chạy thật vì thiếu dữ liệu, weight hoặc xác nhận;
- **Missing**: chưa có implementation đủ để chạy;
- **Deferred**: proposal ghi optional hoặc dữ liệu chưa cho phép.

Điểm quan trọng: repository hiện mô tả khá đúng kiến trúc, nhưng **chưa có artifact kết quả để
chứng minh giả thuyết khoa học**. “Có code” không đồng nghĩa “đã chứng minh”.

## 1. Kết luận nhanh

| Nhóm yêu cầu | Trạng thái sau audit | Bằng chứng chính |
|---|---|---|
| Staged learning: public FM → CTPA DAPT → image-report C0 | Implemented, blocked | `source/pretraining/`, `repr.dapt.*`, `repr.align` |
| Một shared encoder, global + masked heart/PA/lung pooling | Implemented | `source/components/anatomy.py`, `source/components/roi/pooling.py` |
| Organ experts + concat/late-logit/Soft-MoE | Implemented | `source/components/adapters/organ.py`, `source/components/fusion/` |
| Diagnosis native/silver multitask | Implemented, blocked | `source/tasks/diagnosis/`, `diagnosis_organ_silver` |
| Prognosis image + EHR + PESI, PEFT | Implemented, blocked | `source/tasks/prognosis/`, `source/components/peft/` |
| Multitask prognosis heads trong Figure 3.2 | Implemented, blocked | `prog.anatomy.multitask_moe`, `PrognosisHeads` |
| ROI1–ROI8 + matched-random control | Implemented, segmentation-blocked | `source/roi/registry.py`, `source/roi/random_controls.py` |
| Necessity counterfactual, không retrain | Implemented, checkpoint-blocked | `tools/tasks/counterfactual.py`, `anatomy.remove_*` |
| ROI-only sufficiency + teacher/student KD | Implemented, checkpoint-blocked | `source/distillation/`, `anatomy.*_student*` |
| Prognosis organ-only (strict-heart / artery / lung) | Implemented, blocked | `configs/runs/04_prognosis/students/`, `prog.*_student` |
| Sáu architecture ablation | Implemented, blocked | `configs/runs/05_anatomy_analysis/architecture/` |
| Bốn EHR ablation | Implemented, EHR-blocked | `configs/runs/04_prognosis/ehr_ablation/` |
| Patient CV, bootstrap, calibration, paired test | Implemented, data-blocked | `tools/tasks/train_cv.py`, `source/metrics/` |
| PENet-style non-FM baseline | Implemented, data-blocked | `diag.baseline.penet_style`, `source/components/encoders/image/penet.py` |
| Turkey external test-only | Implemented as contract, data-blocked | `diag.external.turkey_test`, `data.{segmentation,roi}.turkey` |
| CTPA-specific nnU-Net fine-tune + expert validation | Missing | Chưa có annotation manifest/model/trainer thật |

## 2. Kiến trúc chính có đúng proposal không?

Có, ở mức implementation.

```text
full 3D CTPA
    │
    └─ shared image encoder, đúng một forward
           └─ spatial feature map F
                ├─ global average pooling ─ global expert
                ├─ F × resized heart mask ─ ROI-normalized pooling ─ heart expert
                ├─ F × resized PA mask    ─ ROI-normalized pooling ─ PA expert
                └─ F × resized lung mask  ─ ROI-normalized pooling ─ lung expert
                                      │
                         concat / late-logit / Soft-MoE
                                      │
                    diagnosis heads hoặc prognosis heads
```

`extract_anatomy_features()` gọi `image_encoder.forward_features(volume)` đúng một lần.
`mask_guided_pool()` resize mask bằng nearest-neighbor, nhân với feature map và chia cho tổng
mask. Vì feature map được sinh từ toàn volume, `z_heart/z_pa/z_lung` là **region-pooled
features**, không được diễn giải quá mức là thông tin chỉ đến từ ROI.

Diagnosis không nhận EHR/PESI. Prognosis mới thêm EHR và PESI vào fusion, đúng email. Nhánh
prognosis hỗ trợ một primary head và nhiều auxiliary native-outcome heads với masked loss;
config chuẩn là `prog.anatomy.multitask_moe`.

## 3. Staged training và checkpoint

| Stage trong email/proposal | Repo | Trạng thái thực tế |
|---|---|---|
| Public CT-FM / CT-CLIP / TotalFM initialization | backbone registry + strict adapter contract | Blocked: ba repo/weight chưa được stage, factory/output adapter/feature dimension còn rỗng |
| Self-supervised CTPA pretraining | none/MAE/DINO/SimCLR/anatomy DAPT | Code có; chưa chạy vì thiếu backbone/data |
| Image-report alignment tạo C0 | symmetric InfoNCE | Code có; report embedding columns chưa được xác nhận |
| Diagnosis từ C0 | single/native multitask/silver/anatomy | Code có; chưa có C0/mask/silver artifacts |
| Prognosis từ C0 hoặc C_diag, dùng PEFT | named initialization + LoRA/frozen/full | Code có; EHR/PESI và event cohort còn blocked |

Checkpoint có lineage, SHA-256, load report và shape-mismatch checks. Không được đổi tên một
checkpoint C0 thành C_diag: source experiment và initialization được lưu riêng.

## 4. ROI và ba họ ablation

| ROI | Định nghĩa code | Vai trò |
|---|---|---|
| ROI1 | heart ∪ mediastinum | keep-only |
| ROI2 | strict heart | keep-only |
| ROI3 | central PA được giãn 2 mm để xóa gần đủ trunk/main PA | remove |
| ROI4 | central PA ∪ intrapulmonary arteries | keep-only |
| ROI5 | whole lung, giữ vessels | keep-only |
| ROI6 | lung trừ arteries/veins lớn | keep-only |
| ROI7 | heart ∪ hilar vessels | remove |
| ROI8 | rigid-translated control cho ROI2/4/6, cùng voxel/physical volume | control |

ROI8 không phải crop hộp chữ nhật tùy ý: code giữ nguyên hình mask bằng rigid translation,
ép nằm trong body, tránh anatomy cấm và giữ z-range mặc định. Không tìm được vị trí hợp lệ thì
case fail thay vì co mask hoặc chọn voxel gần nhất.

Ba câu hỏi được tách đúng:

1. **Architecture contribution**: retrain sáu variants từ C0;
2. **Necessity**: frozen full model, erase ROI trên cùng scan, reuse original masks, không train;
3. **Sufficiency**: student chỉ thấy ROI, init từ C0, train GT-only hoặc GT+KD từ full teacher.

Counterfactual output giữ original/counterfactual probability theo cùng patient và dùng paired
patient bootstrap. Strong ROI performance chỉ chứng minh predictive information, không tự động
chứng minh cơ chế sinh học nhân quả.

## 5. Diagnosis và silver supervision

Repo có report-only, full-image, FM-initialized, PENet-style, native multitask, silver multitask
và anatomy-aware variants. PENet-style mới là project-native residual 3D CNN train end-to-end
từ random weights; nó là **comparison baseline inspired by PENet**, không phải tuyên bố tái tạo
nguyên bản paper PENet.

Silver schema bao phủ PE presence; acuity; central/lobar/segmental/subsegmental/saddle; RV
enlargement/RV-LV/septal bowing/reflux; pleural/pericardial effusion; malignancy; chronic lung
disease/fibrosis/emphysema. Nhãn không chắc chắn abstain, và chỉ accepted rows vào loss.
Native gold labels vẫn là nguồn đánh giá chính.

## 6. Prognosis

Primary protocol là 1-month/30-day mortality trong confirmed acute-PE cohort. All-comers
30-day mortality là secondary analysis; 6/12-month mortality, readmission và 12-month PH là
auxiliary/exploratory. Repo giữ censor/missing là invalid-mask, không biến thành negative.

Các modality baseline có PESI, EHR, image, image+EHR, image+PESI và image+EHR+PESI. Bốn EHR
arms tách all/common variables × with/without missingness indicators. Tất cả prognosis runs
dùng được full/frozen/LoRA transfer, nhưng chạy thật còn chờ:

- danh sách EHR columns và common-variable subset;
- mapping PESI/sPESI được clinical steward duyệt;
- xác nhận cohort/event counts và temporal availability.

## 7. Evaluation protocol

Repo thực thi patient-level split audit, stratified three-fold CV runner, temporal-holdout audit,
AUROC/AUPRC/sensitivity/specificity/F1, prognosis Brier + calibration slope/intercept + curve,
patient bootstrap CI và paired bootstrap comparison.

External test đã được sửa để **không chọn threshold trên external validation**. Turkey/RSPECT
test-only lấy threshold đã khóa trong `result.json` của internal INSPECT validation rồi chỉ đọc
external `test` rows. `train_task.py` từ chối external config.

`train_cv.py` hiện là outer K-fold với một inner validation fold cho mỗi outer test fold và có
ghi fallback reason nếu stratification không khả thi. Nó chưa phải một hyperparameter-search
nested-CV engine hoàn chỉnh; nếu paper tuyên bố “nested CV”, cần thêm search space và inner-fold
selection rõ ràng. Gọi hiện tại là patient-level stratified K-fold CV là chính xác nhất.

## 8. INSPECT, RSPECT và Turkey

- INSPECT: development/internal evaluation; email ước tính khoảng 23K CT và có CT + report +
  EHR/diagnosis/prognosis labels. Repo không hard-code số này thành kết quả cohort; dataset build
  phải ghi số thực sau QC.
- RSPECT/RSNA-STR: email gợi ý là nguồn public hữu ích cho pretraining/supervised transfer.
  Repo giữ nó ở nhánh `repr.rspect.*`; test-only evaluation chỉ là phân tích bổ sung.
- Turkey: khoảng 700 scan theo email nhưng chưa xác nhận; đây mới là primary external test trong
  proposal. Repo đã có segmentation/ROI/test-only contract, nhưng không bịa path, count hay label
  mapping khi data chưa được giao.

Prognosis external Turkey chỉ nên thêm khi Turkey thực sự có outcome + clinical variables; hiện
chưa đủ thông tin để tạo một config trung thực.

## 9. Những gì chưa thể gọi là hoàn thành

1. Ba public backbone chưa tích hợp được vì local third-party directories/weights hiện trống và
   adapter contracts chưa được điền.
2. Chưa có output training/evaluation thật; vì vậy chưa có AUROC/AUPRC/CI để chứng minh proposal.
3. CTPA-specific nnU-Net fine-tuning/freeze chưa implement và thiếu expert-reviewed annotation.
4. EHR/PESI contract chưa được clinical approval; prognosis multimodal chưa chạy được.
5. Turkey chưa có manifest/data/schema xác nhận; external claim chưa thể báo.
6. Report embedding columns và local Falcon/MedGemma weights chưa sẵn sàng.
7. Optional concept-bottleneck có code/config deferred nhưng chưa được xem là primary model.

## 10. Thứ tự cần làm để tạo bằng chứng cho paper

1. Stage INSPECT data và chạy dataset QC để chốt cohort/count/split/event table.
2. Chọn **một** backbone chính, điền factory/output adapter/feature dimension từ checkpoint thật.
3. Stage TotalSegmentator/LungMask weights; tạo mask/ROI và expert-review subset.
4. Chạy PENet-style baseline, public-FM probe, DAPT, alignment C0 và diagnosis baselines bằng cùng CV.
5. Sinh/adjudicate silver labels; chạy native-vs-silver và anatomy architecture ablations.
6. Chốt EHR columns/PESI; chạy prognosis primary cohort và modality/architecture comparisons.
7. Chạy frozen counterfactual + ROI students (diagnosis và prognosis) trên cùng folds/patients.
8. Nhận Turkey data, normalize test-only, tạo masks/ROIs và evaluate checkpoint đã khóa.
9. Chỉ sau đó mới điền các bảng `docs/EXPERIMENTS.md` và viết claim cho MedIA/TMI.

Các lệnh kiểm tra nhanh:

```bash
python run.py list
python run.py show diag.baseline.penet_style
python run.py show prog.anatomy.multitask_moe
python run.py show diag.external.turkey_test
python run.py plan diag.anatomy.silver.moe
```
