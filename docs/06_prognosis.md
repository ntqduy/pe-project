# 06. Quy trình prognosis

## Mục tiêu

Dự đoán `mortality_30d` từ image, EHR và/hoặc PESI.

## Các baseline

- `prog.ehr_pesi`: tabular clinical + PESI, không image.
- `prog.image`: image-only.
- `prog.image_ehr`: image + EHR.
- `prog.image_pesi`: image + PESI.
- `prog.image_ehr_pesi`: image + EHR + PESI.
- Anatomy/global variants: thêm nhánh anatomy khi config yêu cầu.

## Luồng xử lý

1. Đọc prognosis manifest và primary cohort.
2. Fit preprocessing chỉ trên train split: imputation/normalization.
3. Load image encoder từ C0 nếu có image.
4. Load EHR 32 chiều nếu dùng EHR.
5. Load PESI 2 chiều nếu dùng PESI.
6. Fusion các modality, train target mortality, đánh giá calibration/survival metrics.

## Checklist lỗi

- `ehr_columns` phải có đúng 32 biến và không chứa thông tin sau thời điểm dự báo.
- `pesi_features.csv` chỉ được dùng khi clinical approval hoàn tất.
- Imputer/normalizer chỉ fit trên train, không fit toàn bộ cohort.
- Không để primary target hoặc outcome tương lai lọt vào EHR features.
- So sánh image với EHR/PESI phải dùng cùng cohort và split.
