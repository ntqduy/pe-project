# Third-party assets

```text
third_party/
├── repos/         upstream source dạng git submodule (.gitmodules ghim commit)
├── weights/       model artifacts local
├── versions.yaml  URL, pinned commit, expected weight, checksum và trạng thái
└── README.md
```

Không chép project code vào `repos/` hoặc `weights/`, và không sửa upstream source để chứa logic riêng của
project. Adapter tích hợp phải nằm trong `source/components/` hoặc module project tương ứng.

Mỗi repo trong `repos/` là một git submodule: parent project chỉ lưu URL và commit ghim, không chép
source. Máy mới: `git clone --recurse-submodules <repo>` hoặc `git submodule update --init` (một repo:
`git submodule update --init third_party/repos/<tên>`). Thêm repo mới: `git submodule add <url>
third_party/repos/<tên>`. Weights lớn vẫn bị parent project ignore và không lên GitHub. `versions.yaml` là source of truth cần commit: mỗi
component phải ghi repo URL/commit, expected weight path, revision/model ID và SHA-256 khi đã xác minh.

Support pipeline dùng TotalSegmentator làm nguồn pseudo-anatomy canonical. LungMask là model QC độc lập;
Dice thấp có thể flag case nhưng không thay mask lung của TotalSegmentator. CT-FM, CT-CLIP và TotalFM là các
candidate foundation backbone.

Không đổi status thành `ready` trước khi kiểm tra bằng checkpoint thật:

- checksum và exact model/revision;
- factory import path;
- input preprocessing/convention;
- output adapter và feature dimensions;
- khả năng strict-load theo config.

Các field `factory`, `output_adapter`, `feature_dim` đang trống trong `configs/backbone/*.yaml` là blocker có
chủ đích. Không đoán giá trị từ tên repo.
