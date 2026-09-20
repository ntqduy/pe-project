# Local model weights

Model files lớn bị Git ignore. Lưu theo đúng layout mà config hiện tại tham chiếu:

```text
third_party/weights/
  segmentation/
    totalsegmentator/  # toàn bộ offline TotalSegmentator task weights   [MISSING]
    lungmask/R231.pth  # exact LungMask QC checkpoint                    [MISSING]
  ct_fm_feature_extractor/  # pinned clone of the HF weight repo             [STAGED]
    config.json
    model.safetensors
  foundation/
    ct_clip/model.pt                                                     [MISSING]
    totalfm/totalfm_en_checkpoint_best_loss.pt                           [MISSING]
  falcon-7b/         # tiiuae/falcon-7b, complete HF directory           [STAGED]
  medgemma/          # google/medgemma-1.5-4b-it, complete HF directory  [STAGED]
```

Trạng thái hiện tại (kiểm tra trực tiếp từ file trên đĩa, không phải suy đoán):

| model | path | trạng thái |
|---|---|---|
| Falcon | `third_party/weights/falcon-7b` | staged — `FalconForCausalLM`, `tiiuae/falcon-7b` |
| MedGemma | `third_party/weights/medgemma` | staged — `Gemma3ForConditionalGeneration`, `google/medgemma-1.5-4b-it` |
| CT-FM | `third_party/weights/ct_fm_feature_extractor` | staged — SegResEncoder, `project-lighter/ct_fm_feature_extractor` |
| CT-CLIP / TotalFM | `foundation/*` | thiếu weight, thiếu adapter contract |
| TotalSegmentator / LungMask | `segmentation/*` | thiếu weight, chưa cài package |

CT-FM không có bản `.ckpt` nào của Lighter trên upstream: weight công khai duy nhất là HuggingFace
`project-lighter/ct_fm_feature_extractor` (safetensors). Repo đó dùng git-lfs, nên clone bằng `git clone`
thuần chỉ cho ra pointer 134 byte; phải cài git-lfs hoặc tải thẳng file từ endpoint `resolve/main` rồi
đối chiếu SHA-256 với `oid` trong pointer. Loader đọc được `.safetensors` qua
`source/components/encoders/image/external.py:load_state_file`.

MedGemma 1.5 là image-text-to-text checkpoint, nên `auto_model_class` phải là
`AutoModelForImageTextToText`; `AutoModelForCausalLM` không resolve được
`Gemma3ForConditionalGeneration`. Report mining chỉ dùng text nên `AutoTokenizer` vẫn đúng.

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
`tokenizer_class` trong preset tương ứng của `configs/components/silver.yaml`; không sửa
cascade chỉ để né integration contract.
