# 05. Diagnosis

## Nhãn native

Protocol matrix dùng đúng ba nhãn INSPECT đã review:

- `pe_positive` <- raw `pe_positive_nlp`;
- `pe_acute` <- raw `pe_acute`;
- `pe_subsegmental` <- raw `pe_subsegmentalonly`.

Giá trị missing/censored được để rỗng và mask khỏi loss, không đổi thành 0. Các target
report-derived khác chỉ là silver auxiliary supervision.

## Mode và loss

`configs/runs/03_diagnosis/matrix/single_task.yaml` có `diagnosis.mode=single_task` và
một task được chọn. `multitask.yaml` yêu cầu đúng cả ba task, với `pe_positive` là primary.
Config validator kiểm tra `diagnosis.tasks`, `data.label_columns` và model primary head
khớp nhau.

Mỗi head binary dùng BCE-with-logits trên các row observed. Loss multitask là tổng có
trọng số; hiện không bịa class weights. Nếu cần imbalance weighting, phải tính chỉ từ train
split, lưu vào config/result và chạy lại mọi arm so sánh.

## Fine-tuning và experiment tối thiểu

Diagnosis mặc định là LoRA trên module `attn` và `projection`, rank 8, alpha 16, dropout
0.05. Backbone có `lora_target_modules` trong registry thì dùng target đó thay thế: CT-FM
chỉ có Conv3d nên LoRA bọc conv của `layers.3.blocks` và `layers.4.blocks` (16 conv). Năm so sánh tối thiểu dùng cùng split/preprocessing/head/optimizer:

1. single-task từ published/pretrained;
2. single-task từ C0;
3. single-task từ C_silver;
4. multitask từ C0;
5. multitask từ C_silver.

DAPT và diagnosis checkpoint là các initialization bổ sung được hỗ trợ, không thay năm arm
chính dùng để trả lời câu hỏi silver supervision.

Mỗi run ID phải khác và metadata checkpoint ghi source checkpoint/hash. Không so metric của
run có split hoặc cohort khác.

### Early stopping

`training.early_stopping_patience: 10` (mặc định cho mọi contract trong
`configs/components/training.yaml`): dừng sau 10 epoch liên tiếp không cải thiện metric
validation. Đặt `null` để chạy hết `training.epochs` như trước.

- `best.ckpt` luôn là epoch tốt nhất, không phải epoch cuối — dừng sớm không mất model.
- Quyết định dừng tính từ giá trị đã reduce qua mọi rank, nên DDP không bị lệch ở barrier.
- `result.json` ghi `epochs` (ngân sách cấu hình), `epochs_run` (thực chạy),
  `stopped_early` và `early_stopping_patience`.
- So sánh các encoder stage vẫn hợp lệ vì patience giống nhau cho mọi arm. **Không** chỉnh
  patience riêng cho từng checkpoint — làm vậy là confound đúng biến đang muốn đo.

### CT-FM frozen: pooling và chuẩn hoá feature

Áp cho mọi arm CT-FM frozen (baseline `ct_fm_frozen_*` và anatomy `ct_fm_frozen_anatomy_*`):

- **Nhánh global bỏ ô đệm.** Canvas 120×384×384 được lấp không khí quanh vùng quét (17–75% số
  ô feature trên smoke_30). Nhánh global lấy trung bình có trọng số theo kênh coverage của cache
  (`source/components/anatomy.py`, `global_embedding`), nên ô đệm không vào embedding. Trước đây
  là trung bình đều mọi ô: vector "không khí" gần như giống nhau ở mọi bệnh nhân bị trộn vào theo
  tỷ lệ phụ thuộc kích thước cơ thể. Nhánh tim/PA/phổi pool trong mask nên không bị ảnh hưởng.
- **Z-score từng nhánh** (`organ_adapter.standardize_inputs: true`, đặt trong
  `ct_fm_frozen_diagnosis.yaml` và `ct_fm_frozen_prognosis_all.yaml`, các config khác kế thừa).
  Feature CT-FM đã pool có mức nền chung rất lớn (~99% vector global là phần chung của cohort,
  cosine giữa bệnh nhân ~0.99), nên head nhỏ gần như thấy mọi bệnh nhân như nhau. Trước khi
  train, `train_task.py` đi qua split train một lượt, tính `μ`, `σ` cho từng kênh của từng nhánh
  (chỉ trên các dòng có nhánh đó), rồi head nhận `(x − μ) / σ`. `μ`, `σ` là buffer của
  `organ_adapters.input_standardizer`, nằm trong `best.ckpt`, nên evaluate/preview dùng đúng bộ
  số đó. Tóm tắt (số dòng, median |μ|, median σ) ghi ở `logs.txt` và
  `result.json` → `model.feature_standardization`. Nhánh có dưới 2 dòng train (ví dụ cohort
  PE dương tính của smoke_30 chỉ có 1 bệnh nhân train) không tính được σ nên đi qua không chuẩn
  hoá; tên nhánh ghi ở `identity_branches` và một dòng `WARNING` trong log. Kênh hằng số trên
  train giữ scale 1 thay vì chia cho ~0.
- Checkpoint cũ (không có buffer) không load được vào config đã bật tuỳ chọn và ngược lại (load
  strict), nên không thể trộn nhầm hai kiểu. Run trước thay đổi này không so trực tiếp với run
  sau. Không cần chạy lại preprocessing hay cache CT-FM.
- Smoke_30 (`DX_ctfm_frozen`, 30 epoch): độ lệch chuẩn xác suất giữa bệnh nhân tăng từ
  0.005–0.008 lên 0.056–0.079; prognosis từ 0.0003–0.0009 lên 0.033–0.042. Metric smoke vẫn không
  có ý nghĩa vì quá ít ca.

## Đánh giá

Threshold chỉ chọn trên validation rồi khóa cho test. Report gồm AUROC, AUPRC,
sensitivity, specificity, F1 và patient-level bootstrap CI (chỉ trên test). Bảng metric
nằm ở `epoch_<N>/result.csv`, predictions từng ca ở `epoch_<N>/predictions.csv` (xem
`docs/09_output_reference.md`). Validation/test chỉ dùng native label.

```bash
python run.py preflight diag.matrix.single_task
python run.py run diag.matrix.single_task --gpus 0
python tools/tasks/evaluate.py --config configs/runs/03_diagnosis/matrix/single_task.yaml --allow-full
```

### Preview Grad-CAM

Cuối mỗi lần evaluate, `write_backbone_previews` (`source/engine/task_artifacts.py`) ghi
preview cho 5 study đầu của validation theo thứ tự manifest (không chọn theo đúng/sai) vào
`epoch_<N>/preview/`. Mỗi study có hai file:

- `NN_<patient>_<study>_<TP|TN|FP|FN>.html`: viewer offline, không cần mạng. Thanh trượt đi
  qua mọi mức z model thực sự nhận (run CT-FM cached: 10 lát feature); mỗi mức hiện một
  lát CT tham chiếu ở tâm ô feature cạnh CT + Grad-CAM; chỉnh opacity; chọn hiển thị trilinear hoặc ô feature gốc (nearest);
  montage 8 slice điểm CAM cao nhất (bấm để mở slice); mục kỹ thuật thu gọn.
- `NN_<...>.png`: bản tĩnh gồm header, cùng montage đó và colorbar.

Header ghi patient_id, study_id, target giải thích, ground truth, predicted,
TP/TN/FP/FN, xác suất (giá trị của evaluate, đúng bằng `predictions.csv`), threshold và
cách chọn (threshold do evaluate chọn trên validation; preview không chọn lại), và
`p − threshold`. Mỗi slice ghi chỉ số lát feature đầu vào/tổng (hoặc slice CT input với model ảnh), slice gốc
của NIfTI khi ánh xạ xác định được, điểm CAM,
hạng trong volume và hướng A/P/R/L. Thiếu thông tin nào thì ghi N/A.

Tạo lại preview cho một run đã evaluate, không chạy lại evaluate:

```bash
source scripts/use_gcs_storage.sh
python tools/tasks/gradcam_preview.py --run-dir <output>/diagnosis/<RUN_ID>
# --output <dir> để ghi chỗ khác; --method gradcam để dùng Grad-CAM gốc
```

**Kiến trúc và gradient.** Model là 3D, không gộp slice 2D. Với run `ct_fm_features`, CT-FM
(SegResEncoder) đã chạy lúc build cache trên các patch 3D 24×128×128 không chồng lấn của
canvas 120×384×384 (3×1×1 mm, SPL); output `layers[4].blocks` (512 kênh, 2×8×8/patch) được
ghép thành feature map 512×10×24×24. Đây là target layer: tensor này chính là input của phần
được train, nên gradient của logit `pe_present` lấy trực tiếp qua pooling → adapter → MLP
(CT-FM đã đóng băng). Với backbone chạy trên ảnh, hook lấy output 5D sâu nhất của backbone và
gradient đi qua toàn bộ phần sau nó. Preview chạy `model.eval()` + `torch.enable_grad()`, một
backward, xóa gradient và trả lại mode cũ; không đổi weights, preprocessing hay logic dự đoán.
Forward chạy trên module đã unwrap, không qua DDP: chỉ rank 0 ghi preview, và một forward/backward
DDP ở rank 0 sẽ mở collective mà các rank khác không tham gia (code cũ bị treo ở trường hợp này).

**Phương pháp.** Mặc định là Grad-CAM dạng element-wise (HiResCAM):
`CAM = ReLU(Σ_k ∂y/∂A_k ⊙ A_k)`. Khi head pool trung bình đều, gradient giống nhau ở mọi ô và
cách này trùng Grad-CAM gốc (`--method gradcam`, trọng số kênh = trung bình không gian của
gradient); khi pooling có trọng số hoặc theo ROI, nó chỉ gán đóng góp cho ô model thực sự đọc.
Preview đo kiểu pooling từ gradient và ghi vào mục kỹ thuật. Đây là bản đồ đóng góp cho logit,
khác feature activation (độ lớn activation, không phụ thuộc target), và preview không vẽ
activation.

**Hiển thị.** CAM dương được chia cho max của nó trên lưới feature của cả volume: một thang
màu (inferno) và một colorbar cho mọi slice và montage, không chuẩn hóa từng slice. Overlay có alpha = opacity × CAM đã chuẩn hóa; nơi CAM bằng 0 giữ nguyên CT. Không dùng
ngưỡng hay mask cơ thể/phổi, nên tín hiệu ở không khí ngoài cơ thể và padding vẫn hiện ra.
CT dùng W/L 700/100 HU. Với CT-FM cached, điểm slice là trung bình CAM dương của từng lát
feature sau nội suy bilinear trong mặt phẳng, trước chuẩn hóa hiển thị; với model nhận CT,
điểm được tính sau nội suy trilinear `align_corners=False`; montage lấy 8 slice điểm cao nhất (hòa thì z nhỏ trước). Montage chọn
theo CAM, không phải slice đã xác nhận có PE. CAM toàn 0 hoặc có NaN/Inf: không vẽ overlay,
không xếp hạng, không montage, header ghi lý do; xác suất NaN thì không gán nhãn dự đoán.

**Căn chỉnh.** Với run cached, CT hiển thị là canvas dựng lại từ NIfTI gốc bằng spec trong
sidecar; preview so shape/affine/bounds với sidecar cache và cảnh báo nếu khác. Các kiểm tra
một lần trên `DX_ctfm_frozen__ds_smoke_30`:

| Kiểm tra | Kết quả |
|---|---|
| CT-FM chạy lại trên canvas dựng lại vs feature cache | corr 0.99999998, sai số tương đối 3.6e-4 (float16) |
| Lưới feature trong sidecar vs quy ước `F.interpolate(align_corners=False)` | trùng (ô i phủ voxel [12i, 12i+12)) |
| Dựng lại slice canvas từ NIfTI gốc bằng chuỗi ánh xạ slice gốc (5 ca, cả 3 trục) | sai số 0.0 |
| Cùng phép dựng nhưng qua `output_affine` của sidecar | lệch trung bình ~9–80 HU; theo z affine trôi tới 0.9–3 mm |
| Pixel overlay trong Chromium vs `torch` trilinear | ≤ 1.2 mức xám (làm tròn bin màu) |
| Xác suất preview vs evaluate; weights trước/sau | Δ = 0.0; hash state_dict không đổi |
| DDP 2 tiến trình (gloo), rank 0 ghi preview, rank 1 chờ barrier | xong trong 8 s; code cũ treo tới timeout |

Slice gốc được tính theo đúng chuỗi preprocessing (canvas crop/pad → body crop →
`ndimage.zoom` → đảo trục), không theo `output_affine`. `_affine_with_spacing` giả định zoom là
phép co giãn spacing thuần, còn `ndimage.zoom` khớp tâm voxel đầu và cuối, nên affine trôi dần
theo z: từ 0 ở slice đầu tới 0.9–3 mm (≤ 1 slice canvas) ở đầu kia trên 5 ca smoke. Mask ROI được
đặt lên lưới bằng affine này (`source/imaging/grid.py`) nên chịu cùng độ lệch; preview chỉ báo,
không sửa preprocessing. Mỗi slice canvas là nội suy tuyến tính giữa hai slice gốc. Viewer cached chỉ hiển thị
10 lát CT tham chiếu tại tâm các ô feature z, kèm ánh xạ các lát đó về NIfTI gốc;
lát nằm trong padding của canvas ghi "padding" thay vì một chỉ số.

**Giới hạn cần biết khi đọc preview.**

- Độ phân giải thật của CAM là một ô feature 12×16×16 voxel (36×16×16 mm). Nội suy bilinear trong mặt phẳng ở viewer cached chỉ
  làm mượt, không tăng độ chính xác định vị. Đo gradient trên một patch cho thấy mỗi ô chỉ lấy
  ~5–12% độ nhạy từ block 12×16×16 của nó, và hai ô z của cùng một patch 24 slice nhìn gần như
  cùng vùng (~50/50), nên vị trí thật còn thô hơn lưới ô.
- Nhánh global dùng trung bình có trọng số body-coverage (xem "CT-FM frozen: pooling và chuẩn
  hoá feature"), nên ô padding của canvas không có gradient; preview vẫn ghi % CAM dương nằm
  trong padding như một QC. Phần còn lại (~5% trên smoke_30) đến từ nội suy trilinear ở rìa khi
  phóng CAM lên lưới hiển thị. Header ghi "pooling có trọng số" khi ∂logit/∂A không đồng nhất.
- Trước khi sửa pooling (run smoke `DX_ctfm_frozen__ds_smoke_30` cũ, 11 ca train, mọi xác suất
  ~0.28), CAM gần như phẳng theo z: 41–58% khối CAM dương nằm trong padding canvas, 53–66% nằm
  ngoài mask cơ thể thô, và các ô z chẵn (nửa đầu mỗi patch 24 slice) mang 87–92% khối CAM. Đó
  là dấu hiệu model chưa định vị gì và có artifact theo patch, không phải giải phẫu.
- Montage chỉ lấy đúng 8 lát feature có điểm cao nhất, nên có thể gồm các lát padding.

**PENet zero-shot.** `tools/tasks/zeroshot_penet.py` ghi cùng loại preview (`source/imaging/penet_preview.py`)
vào `RUN/preview/`. PENet là CNN 3D chạy trên từng cửa sổ 32 slice không chồng lấn
(1×32×208×208 sau resize 224 INTER_AREA + crop 208), rồi gộp xác suất các cửa sổ (mặc định max).
Preview chạy lại model đã load, giải thích y = logit(xác suất series):

- target layer `encoders[-1]` (2048×2×7×7 mỗi cửa sổ, ngay trước `GAPLinear`); vì GAP nên
  HiResCAM = Grad-CAM gốc;
- gradient qua bước gộp: với max, ∂y/∂logit_w = 1 ở cửa sổ quyết định và 0 ở mọi cửa sổ khác, nên
  chỉ cửa sổ đó có CAM; với mean, ∂y/∂logit_w = p_w(1 − p_w)/(n·p̄(1 − p̄));
- thanh trượt đi qua mọi slice PENet nhận (gồm slice không khí đệm cho đủ bội số 32), mỗi slice ghi
  cửa sổ, xác suất cửa sổ và slice gốc (1:1 vì PENet không resample theo z); slice đầu vào 1 là
  phía trên với `slice_order=superior_to_inferior`;
- lưới CAM chỉ 7×7 trong mặt phẳng (208/7 không chia hết), mỗi ô ~16 slice × ~30 pixel; nội suy
  trilinear làm trong từng cửa sổ, không qua ranh giới cửa sổ.

Kiểm tra trên `DX_zeroshot_penet__ds_smoke_30`: input của preview trùng từng byte
`series_windows`; 239/239 slice ánh xạ đúng giá trị slice gốc; CAM qua bước gộp max và mean trùng
autograd đầy đủ qua cả hai cửa sổ (lệch tương đối ~1e-7); xác suất preview lệch inference 3.7e-8;
chạy lại tool cho `predictions.csv`/`result.csv` giống hệt; weights không đổi. Tạo lại cho run có sẵn:

```bash
python -m source.imaging.penet_preview --run-dir <output>/diagnosis/DX_zeroshot_penet__ds_<profile> \
  --raw-root /mnt/Stanford_INSPECT_dataset/CT/full/CTPA
```

**RADAR zero-shot.** `tools/tasks/zeroshot_radar.py` (env project) chọn study và đánh giá như
PENet; model chạy trong env riêng của RADAR qua `tools/tasks/zeroshot_radar_worker.py`, tái hiện
`RADAR_inference/inference_demo.py` cho một finding: resample 1×1×5 mm, clip [-300, 400] HU,
cửa sổ 96×256×384, đầu phân đoạn của RADAR tự tìm động mạch phổi (`肺动脉`), rồi so token cơ quan
đó với cặp câu `radar.prompts` (âm: `normal.`, dương: trung bình các câu kiểu báo cáo về tắc mạch
phổi, do project viết vì RADAR không có finding PE). Preview (`RUN/preview/*.png`, 5 ca validation
đầu) vẽ tensor đầu vào của model ở lát axial/coronal có nhiều voxel động mạch phổi nhất, phủ mask
động mạch phổi của chính RADAR; tiêu đề ghi GT, pred, p và ngưỡng. Không có CAM: điểm số đến từ
attention của token cơ quan, không từ một bản đồ không gian. Study mà RADAR không tìm thấy động
mạch phổi bị bỏ qua và liệt kê trong `result.json` (`skipped_series`). Lệnh chạy: xem
`scripts/README.md` (Zero-shot RADAR).

### Chọn encoder stage bằng `ENCODER_SOURCE`

Đây là biến quyết định cả chương trình so sánh: nó chọn checkpoint mà encoder khởi tạo từ đó,
và được stamp vào run ID (`__enc_dapt`, `__enc_silver`) nên các lần chạy không đè nhau.

```bash
for SRC in pretrained dapt c0 silver; do
  # fine-tune, single-task
  python tools/tasks/train_task.py --config configs/runs/03_diagnosis/matrix/single_task.yaml --set data.profile=full_inspect --set encoder.init_source=$SRC --gpus 0
  # probe, single-task và 3 nhãn
  python run.py run probe.diag --set encoder.init_source=$SRC
  python run.py run probe.diag.multitask --set encoder.init_source=$SRC
done

# fine-tune, 3 nhãn
for SRC in c0 silver; do
  python tools/tasks/train_task.py --config configs/runs/03_diagnosis/matrix/multitask.yaml --set data.profile=full_inspect --set encoder.init_source=$SRC --gpus 0
done
```

Giá trị hợp lệ: `pretrained` (alias `published` / `public` / `original`), `dapt`,
`c0` (alias `image_report` / `alignment`), `silver`, `diagnosis`, `rspect_multitask`,
`rspect_single`, `custom`. Với `custom` phải kèm `ENCODER_CHECKPOINT` và
`ENCODER_EXPERIMENT` để provenance vẫn được ghi vào lineage.

Chạy cả probe lẫn fine-tune rồi báo cáo cả hai: thứ hạng ở hai nhánh có thể lệch nhau, và
đó là kết quả đáng ghi. Kế hoạch experiment đầy đủ: [EXPERIMENTS.md](EXPERIMENTS.md).

Silver input mới là `silver_labels.csv`, chỉ row `label_status=accepted`/legacy
`status=accepted` được dùng.
