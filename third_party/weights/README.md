# Local model weights

Model files lớn bị Git ignore. Lưu theo đúng layout mà config hiện tại tham chiếu:

```text
third_party/weights/
  segmentation/
    totalsegmentator/  # toàn bộ offline TotalSegmentator task weights
    lungmask/R231.pth  # exact LungMask QC checkpoint
  foundation/
    ct_fm/model.ckpt
    ct_clip/model.pt
    totalfm/totalfm_en_checkpoint_best_loss.pt
  silver/
    falcon/       # complete local Hugging Face model directory
    medgemma/     # complete local Hugging Face model directory
```

Segmentation preflight yêu cầu cả local CLI installation và các offline path này. Project không tự download
checkpoint. TotalSegmentator là nguồn mask chính; LungMask chỉ đóng góp lung Dice QC.

Với Hugging Face directory, giữ mọi file mà `from_pretrained` cần: model config, tokenizer/processor, weight
shards và shard index. Silver generation mặc định `local_files_only: true`, nên không âm thầm tải model khác.

Sau khi thêm/thay weight:

1. tính SHA-256;
2. ghi exact model/revision/checksum vào `third_party/versions.yaml`;
3. điền adapter contract tương ứng nếu là backbone;
4. chạy preflight cho experiment dùng weight đó.

Nếu silver model không tương thích `AutoModelForCausalLM`/`AutoTokenizer`, đổi `auto_model_class` hoặc
`tokenizer_class` trong `configs/silver/*.yaml`; không sửa cascade chỉ để né integration contract.
