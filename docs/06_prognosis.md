# 06. Prognosis

## Cohort

- `all_comers`: một index CTPA sớm nhất cho mỗi patient đủ điều kiện chung.
- `pe_positive_only`: một index CTPA dương tính PE sớm nhất cho mỗi patient.

Để tương thích, file vẫn tên `prognosis_all_patient.csv` và
`prognosis_pe_positive.csv`; launcher vẫn nhận aliases `all_patient`, `PE_positive` nhưng
chuẩn hóa tên run mới. `prognosis_cohort_membership.csv` ghi từng patient/cohort:
`patient_id, study_id, reference_study_id, split, cohort_name, cohort_config_key,
pe_status, inclusion_status, exclusion_reason`.

## Bảy endpoint

1/6/12-month all-cause mortality, 1/6/12-month readmission và 12-month pulmonary
hypertension theo cột INSPECT. Code không có bằng chứng để gọi mortality là PE-related,
vì vậy docs không đổi nghĩa endpoint. Mỗi endpoint có event, status, observed, censored,
time-to-event và source censor flag. Censored/missing không được coi là negative.

## Cửa sổ và modality

`EHR_0_h` dùng dữ liệu trước index time; `EHR_24_h` áp buffer nghiêm ngặt hơn theo định
nghĩa trong `<dataset_root>/clinical/ehr_profiles.json` (artifact do stage 0 sinh ra,
không phải file trong `configs/`). Imputation và normalization chỉ fit trên
train. Các arm image, EHR, PESI và fusion phải dùng cùng cohort/split/endpoint.

Image encoder có thể init từ pretrained, DAPT, C0 hoặc diagnosis checkpoint. Diagnosis
init đã là source hợp lệ và transfer chỉ `image_encoder`; prognosis head luôn mới.

## Chạy matrix

Các file dưới `scripts/5_prognosis/<cohort>/<ehr_profile>/<task>/` là **symlink** tới
`_strategy.sh`, và `_strategy.sh` suy `COHORT`/`EHR_PROFILE`/`TASK`/`STRATEGY` từ **đường dẫn
được gọi** — nên gọi thẳng `_strategy.sh` sẽ suy sai và bị từ chối.

```bash
# đúng: gọi qua symlink, đường dẫn mang thông tin
bash scripts/5_prognosis/all_patient/EHR_0_h/1_month_mortality/image_only.sh
```

Trên Windows các symlink này thường bị checkout thành file text 21 byte và không chạy được.
Khắc phục bằng `git config core.symlinks true && git checkout -- scripts/5_prognosis`
(cần Developer Mode), hoặc gọi trực tiếp qua matrix runner:

```bash
COHORT=all_patient EHR_PROFILE=EHR_0_h TASK=1_month_mortality STRATEGY=image_only \
  bash -c 'source scripts/_matrix_runner.sh && matrix_run_prognosis'
```

Metric test kèm CI chỉ có sau khi chạy `evaluate.py`:

```bash
python tools/tasks/evaluate.py --config configs/runs/04_prognosis/modality/image.yaml --allow-full
```

Mỗi tổ hợp cohort x EHR window x endpoint x encoder source x strategy có output ID riêng.
Đánh giá chọn threshold trên validation, báo AUROC/AUPRC, sensitivity/specificity/F1,
calibration/Brier khi có, và patient bootstrap CI. Không tune trên test.

## Organ-only sufficiency (heart / PA / lung)

Email của giáo liệt kê trong "Summary of prognosis experiments" ba arm riêng: *strict-heart
prognosis model*, *artery-only prognosis model*, *lung-only prognosis model*. Đây là bản
prognosis của `anatomy.*_student` bên diagnosis.

| Experiment | ROI | Câu hỏi |
|---|---|---|
| `prog.heart_student` | ROI2 strict heart | Riêng vùng tim có đủ dự đoán tử vong 30 ngày? |
| `prog.pa_student` | ROI4 PA tree | Riêng vùng huyết khối có đủ? |
| `prog.lung_student` | ROI6 lung parenchyma | Riêng nhu mô phổi có đủ? |
| `prog.random_student` | ROI8 matched random | Sàn đối chứng — ba arm trên chỉ có nghĩa khi vượt arm này |

Ba điểm thiết kế, đều cố ý:

- **`task.regions: []`** — kiến trúc giống hệt `prog.image` (global encoder, không nhánh
  organ). Hàng tham chiếu là `prog.image`, nên giữa hai hàng chỉ khác *input*, không khác
  kiến trúc. Nếu student mang nhánh heart/PA/lung còn `prog.image` thì không, chênh lệch
  giữa hai hàng sẽ trộn "vùng nào" với "kiến trúc nào". Muốn so với `prog.anatomy.*` thì
  `--set task.regions=[heart,pa,lung]`, nhưng khi đó arm không còn ghép cặp với `prog.image`.
- **`task.modalities: [image]`** — không EHR, không PESI. Claim là *riêng vùng này* mang tín
  hiệu tiên lượng, nên nhánh lâm sàng phải tắt. Kèm lợi ích thực tế: arm không thừa hưởng
  blocker `ehr_columns` lẫn blocker clinical-approval của PESI.
- **Không KD.** `source/engine/task_steps.py` chỉ nối frozen teacher cho stage diagnosis; bật
  `distillation.enabled` ở đây sẽ raise. Bốn arm đều là ground-truth-only.

Mỗi arm là `prog.image` với toàn bộ voxel ngoài ROI bị thay bằng policy `local_mean` dùng
chung — cùng backbone, cùng init C0, cùng LoRA contract, cùng head, cùng cohort/split.

```bash
# cả bốn arm, train + evaluate, tuần tự
bash scripts/run_prognosis_organ.sh

ARMS="heart random" bash scripts/run_prognosis_organ.sh    # chỉ một phần
ACTION=preflight bash scripts/run_prognosis_organ.sh       # kiểm tra, không train
CV=1 bash scripts/run_prognosis_organ.sh                   # K-fold thay hold-out

# từng arm
bash scripts/5_prognosis/heart_student.sh
bash scripts/5_prognosis/pa_student.sh
bash scripts/5_prognosis/lung_student.sh
bash scripts/5_prognosis/random_student.sh

# hàng trần để đối chiếu
bash scripts/5_prognosis/image_only.sh
```

Bốn arm cũng có mặt trong protocol matrix với `STRATEGY=heart_student|pa_student|
lung_student|random_student`, nên chạy được trên cả 7 endpoint và cả hai cohort:

```bash
COHORT=PE_positive EHR_PROFILE=EHR_0_h TASK=1_month_mortality STRATEGY=pa_student \
  bash -c 'source scripts/_matrix_runner.sh && matrix_run_prognosis'
```

Output: `<output_root>/roi_students/prognosis/RS_prog_<arm>_only_gt/` — tách khỏi
`roi_students/` của diagnosis vì hai nhóm có hàng tham chiếu khác nhau và không bao giờ
được đọc chung một bảng.
