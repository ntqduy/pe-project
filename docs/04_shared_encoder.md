# 04. Quy trình shared encoder

## Ý tưởng

Tạo một representation chung rồi chuyển trọng số image encoder sang các task downstream. Chuỗi chính:

`public CT backbone → DAPT (tùy chọn) → image-report alignment → C0 → diagnosis/prognosis/ROI students`.

## Các stage

### DAPT

Config ở `configs/runs/02_representation/dapt/`. Method gồm `none`, MAE, DINO, SimCLR và Anatomy-DAPT. `none` chỉ chuẩn hóa initialization và ghi checkpoint.

### Image-report alignment

`tools/pretrain_model/train_alignment.py` dùng precomputed report embeddings, không chạy text encoder. Loss là symmetric InfoNCE; batch size phải `>= 2`. Output chuẩn là:

`outputs/pretraining/alignment/AL_image_report_c0/best.ckpt`.

### Transfer

Downstream dùng `encoder/from_c0.yaml` và transfer module `image_encoder`. PEFT/freeze quyết định phần nào được train tiếp.

## Checklist lỗi

- `alignment.report_embedding_columns` phải chứa tên cột embedding thật trong `paired_reports.csv`.
- Feature dimension của backbone phải khớp checkpoint.
- Checkpoint C0 phải tồn tại trước diagnosis/prognosis.
- Không trộn checkpoint khác backbone hoặc khác preprocessing fingerprint.
- Kiểm tra lineage trong `result.json`, không chỉ kiểm tra file `best.ckpt`.

