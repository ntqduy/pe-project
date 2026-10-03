# 02. Models

**Đọc khi:** cần biết một config biến thành `nn.Module` thế nào; muốn thêm/đổi backbone; muốn hiểu
forward pass (tensor shape) của baseline 3D và 2D/2.5D slice-MIL; muốn biết PEFT (full / frozen /
LoRA) thực sự đóng băng cái gì.

**Code chính:** `source/engine/factory.py`, `source/components/encoders/image/`, `source/model/`
(`base.py`, `classifier.py`, `registry.py`, `inflate.py`, `weights.py`, `2D_model/`, `3D_model/`,
`head/`), `source/components/{peft,adapters/standardization.py}`,
`configs/components/{backbones,encoders,baselines}.yaml`.

Dữ liệu ở [01_data_pipeline.md](01_data_pipeline.md); loss, trainer, evaluation ở
[03_training_evaluation.md](03_training_evaluation.md); các lưới thí nghiệm exp01-exp04 và
zero-shot ở [04_experiments.md](04_experiments.md).

---

## 1. Lắp model: `build_task_model`

`source/engine/factory.py:build_task_model(config) -> (model, peft_report)` là điểm vào duy nhất
(train, evaluate, Grad-CAM preview đều gọi nó).

```text
config  (experiment.stage ∈ {diagnosis, prognosis})
  └─ task.architecture == "baseline_classifier"   (giá trị khác -> ValueError)
        └─ build_image_encoder -> apply_peft -> BaselineClassifier(encoder, targets, head)
```

- **Một kiến trúc trainable duy nhất**: `baseline_classifier` (chỉ dùng `global_embedding` của
  encoder). Zero-shot PENet/RADAR không đi qua factory (runner riêng, [04](04_experiments.md)).
- PEFT chỉ áp lên **image encoder**; standardizer, projection, head luôn trainable/fit.
- Prognosis chỉ image-only (`task.modalities` phải là `[image]`). Với arm CT-FM, cờ
  `--scratch` dịch thành `model.load_pretrained=false`; với `cached_features` thì scratch bị cấm
  (feature đã trích bằng weight pretrained).
- Chọn backbone: `--set model.backbone=resnet18_3d`; `resolve_backbone` (`source/utils/config.py`)
  copy entry tương ứng trong `configs/components/backbones.yaml#registry` vào `model`. Đổi backbone
  khác preset sẽ stamp `__bb_<backbone>` vào run id.

---

## 2. Hợp đồng encoder: `BaseImageEncoder`

File `source/components/encoders/image/base.py`. Mọi encoder trả `ImageFeatures`:

`ImageFeatures`: `feature_map [B, C_map, d, h, w]` (luôn 5-D, theo trục input RAS; Grad-CAM đọc nó), `global_embedding [B, feature_dim]` (luôn 2-D, cái classifier dùng), `pyramid` (level trung gian), `metadata` (`{"backbone", "pooling", ...}`).

`forward(volume)` trả `global_embedding`. `feature_map_dim` (số kênh `feature_map`) mặc định bằng
`feature_dim`; chỉ khác khi embedding không phải mean của map (nnMamba: embedding 448 = concat mean
3 level, map 256).

**`train(mode)` override.** Trainer gọi `model.train()` mỗi epoch, sẽ bật lại train mode cho encoder
đã đóng băng (BatchNorm cập nhật running stats với batch 1, Dropout bật). Override xử lý:

| Tham số encoder | Hành vi |
|---|---|
| Tất cả trainable (`full`, scratch) | giữ train mode |
| Không tham số nào trainable (`frozen`, `linear_probe`) | cả encoder về `eval` |
| Một phần (LoRA) | submodule toàn tham số frozen → `eval`; mọi BatchNorm/InstanceNorm có `track_running_stats` mà affine của chính nó frozen → `eval`. Chỉ LoRA adapter và `peft_trainable_modules` ở train mode |

`apply_peft` gọi lại `module.train(module.training)` ngay sau khi đóng băng để quy tắc có hiệu lực tức thì.

---

## 3. Registry backbone

1. `source/components/encoders/image/registry.py`: `_REGISTRY` tên → builder (tên chuẩn hoá `lower`,
   `-` → `_`). Đăng ký sẵn `ct_fm`, `ct_fm_features` và toàn bộ baseline zoo. Tên lạ →
   `KeyError` liệt kê `registered_backbones()`.
2. `source/model/registry.py:BASELINE_BACKBONES`: `name -> (module, function)` dưới `source.model`,
   import **lazy** (thư mục `2D_model`/`3D_model` không phải identifier hợp lệ; thiếu
   `mamba_ssm`/`monai`/`timm` chỉ làm hỏng arm cần nó).

---

## 4. CT-FM và encoder ngoài

### 4.1 CT-FM image-space (`backbone: ct_fm`)

```text
cache volume [B,1,x(R),y(A),z(S)]  min-max [0,1] của [-1000,1000] HU
   │ remap_intensity: HU = v·2000 − 1000 → (HU + 1024)/3072, clamp [0,1]
   │ ras_to_spl: permute(0,1,4,3,2).flip(3,4) → [B,1,S,P,L]
   ▼
MONAI SegResEncoder (init_filters 32, blocks_down [1,2,2,4,4]) → pyramid 5 level
   │ spl_to_ras trên MỌI level → feature map trở lại trục RAS
   ▼
feature_map = pyramid[-1]; global = adaptive_avg_pool3d → [B,512]
```

- `CTFMInputContractEncoder` là **subclass** của `SegResEncoder` nên state-dict key giữ nguyên → strict load đủ 161 tensor. Feature map trả về theo RAS (Grad-CAM khớp input). Spacing **không** được chuyển (cache 1.5 mm khác 3×1×1 mm của CT-FM). Input 128³ cho `feature_map [B,512,8,8,8]`.
- LoRA: `SegResEncoder` không có `nn.Linear`, nên `lora_target_modules: [layers.3.blocks, layers.4.blocks]` (conv hai stage sâu nhất).

### 4.2 CT-FM cached features (`backbone: ct_fm_features`)

Nhét cả ngực vào một patch 24×128×128 cho voxel ~12×1.9×2.6 mm là scale CT-FM chưa từng thấy, nên
`tools/data/build_ctfm_cache.py` trích feature **một lần/study** theo contract upstream (xem
[01_data_pipeline.md](01_data_pipeline.md) §1.6).

- Tensor cache `[B, 513, d, h, w]` = 512 kênh feature + 1 kênh "valid" (tỉ lệ ô trong body box).
  `CTFMFeaturePassthrough` chỉ tách hai phần.
- `global_embedding` = **mean có trọng số body** (loại ô padding không khí), `metadata.pooling =
  "body_weighted_mean"`.
- `CachedCTFMEncoder` không có tham số; chỉ dùng frozen, không chạy scratch được. Arm
  `ctfm_frozen_3d` (đường CT-FM frozen duy nhất) đọc bản pooled `[513,1,1,1]`; cache dựng một lần
  mỗi profile bằng `bash scripts/diagnosis/baselines/prepare_ctfm_cache.sh`.

### 4.3 Hai biến thể CT-FM nhận đầu vào khác nhau (có chủ ý)

| | `ctfm_frozen_3d` | `ctfm_lora_3d` |
|---|---|---|
| Nguồn ảnh | NIfTI gốc → cache CT-FM (`build_ctfm_cache.py`) | Volume chung `128³` của mọi baseline 3D |
| Hướng | SPL | SPL (`ras_to_spl` trong encoder) |
| Cường độ | HU clip [-1024, 2048] → [0, 1] | đổi từ [-1000, 1000] sang thang (HU + 1024)/3072 (`remap_intensity`; HU > 1000 đã bị cache cắt) |
| Spacing | **3 × 1 × 1 mm** (đúng pretrain) | **1.5 mm đẳng hướng** |
| Không gian đưa vào encoder | canvas 120×384×384, **45 patch 24×128×128** không chồng lấn | **một khối 128³** cho cả lồng ngực |
| Cái gì được train | chỉ projection + head (feature tính một lần) | LoRA ở `layers.3/4.blocks` + projection + head |

- **Frozen** theo đúng contract pretrain của CT-FM: feature tính một lần nên chi phí không đáng kể.
- **LoRA** dùng volume chung 128³ để giữ chi phí train như các baseline 3D khác. Đưa LoRA về đúng
  contract pretrain cần cache canvas CT-FM (~35 MB/study float16, ~800 GB cho toàn INSPECT) và
  forward + backward qua 45 patch mỗi study ở **mọi epoch** (ước tính vài giờ/epoch trên một L4),
  nên không chọn. Hướng ảnh và thang cường độ vẫn khớp pretrain; chỉ spacing và cách chia patch khác.
- **Khi báo cáo:** chênh lệch frozen vs LoRA gồm cả cách thích nghi (frozen / LoRA) lẫn đầu vào
  (canvas CT-FM vs volume 128³). Ghi rõ điều này trong Methods; đừng diễn giải nó như hiệu ứng thuần
  của LoRA.

### 4.3 PENet

`penet_3d` (`3D_model/penet.py`): PENet thật (`third_party/repos/penet`, weight `penet_best.pth.tar` load strict + SHA-256), RAS → layout DICOM của PENet, intensity `penet`, giữ stochastic depth nên `encoder.ddp_find_unused_parameters = True`. Zero-shot (`penet_zeroshot.py`, `diag.zeroshot.penet`) đọc NIfTI gốc, không train.

---

## 5. Baseline zoo (`source/model/`)

Mọi encoder zoo là `BaselineEncoder` (`base.py`): bọc một *core* trả `{"feature_map",
"global_embedding"?, "pyramid"?, "pooling"?, "metadata"?}`; core giữ ở `self.model` (Grad-CAM hook
`encoder.model`), input đi qua `self.preprocess` (`IntensityAdapter`). Thiếu `global_embedding` →
mean không gian của map.

**Intensity mapping.** Cache là `[B,1,x,y,z]` RAS, [0,1] của [-1000,1000] HU; mỗi encoder đổi về
quy ước lúc pretrain (không tham số nên checkpoint load chéo giữa các mode): `unit` (giữ nguyên),
`window` (clip HU rồi [0,1], thêm `(v-mean)/std`; ImageNet backbone dùng `[-250, 450]`), `zscore`
(z-score từng volume), `penet` (clip [-100,900] → [0,1] − 0.15897). Cấu hình `model.intensity`.

### 5.1 2D / 2.5D slice-MIL (`source/model/2D_model/`)

- `Timm2DBackbone`: `timm.create_model(..., num_classes=0)`, chuẩn hoá output về NCHW; head timm
  thay `nn.Identity` (tránh tham số không gradient dưới DDP).
- `SliceMILCore`: `uniform_slice_indices(depth, N)` chọn N tâm cách đều trên z. `2d` = 1 slice axial
  (1 kênh); `2.5d` = 3 kênh `[i-offset, i, i+offset]`. Resize bilinear về 224, encode theo chunk,
  pooling `attention` (`GatedAttentionPool`), `mean` hoặc `max`. Output `feature_map [B,C,h,w,N]`
  (N slice ở trục cuối), `pooling = "slice_mil_<pooling>"`, metadata `slice_indices`,
  `slice_attention`, `slice_depth`.
- `model.mil.slice_selection`: `uniform` (mặc định, N slice cách đều) hoặc `center` (chỉ 1 instance
  ở lát giữa: 2D = lát giữa, 2.5D = 3 lát giữa); `model.mil.pooling` ∈ `attention | mean | max`.
  Đây là các biến thể của exp04.
- `build_slice_mil_encoder` đặt `peft_trainable_modules = ("model.attention",)`: attention pool là
  module mới không có weight pretrained nên **vẫn train dưới LoRA/frozen**.

Cặp `_2d`/`_25d` dùng cùng builder, khác `slice_mode` trong `backbones.yaml` (32 slice, 224², attention hidden 128).

### 5.2 Bảng baseline (2D/2.5D và 3D)

| backbone | Kiến trúc | Pretrained | Intensity | `feature_dim` | LoRA mặc định |
|---|---|---|---|---|---|
| `{resnet18,convnext,vit,swin}_{2d,25d}` | timm (`resnet18.a1_in1k`, `convnext_tiny`, `vit_small_patch16_224`, `swin_tiny_patch4_window7_224`) | ImageNet | window+ImageNet | 512 / 768 / 384 / 768 | `layer3,layer4` / `stages.3` / `attn.qkv,attn.proj` (vit, swin) |
| `resnet18_3d` / `resnet50_3d` | MONAI ResNet 3D | MedicalNet 23 datasets | zscore | 512 / 2048 | `layer3`, `layer4` |
| `densenet121_3d` | MONAI DenseNet-121 3D | ImageNet, inflate 2D→3D | window+ImageNet | 1024 | `denseblock4` |
| `convnext_3d` | ConvNeXt-T 3D tự viết | ImageNet inflate | window+ImageNet | 768 | `stages.3` |
| `vit_3d` | block ViT-S/16 timm + `Conv3d` patch 16³ | ImageNet inflate (pos table resize, bỏ cls token) | window+ImageNet | 384 | `attn.qkv`, `attn.proj` |
| `swin_3d` | MONAI `SwinTransformer` (encoder Swin-UNETR) | `model_swinvit.pt` SSL trên CT (tự tải) | unit | 768 | `attn.qkv`, `attn.proj` |
| `nnmamba_3d` | nnMamba4cls (`third_party/repos/nnMamba`) | **không có** weight công khai, scratch | unit | 448 (map 256) | `layer3` |
| `mamba_mae_3d` | Vision Mamba 3D (`third_party/repos/mamba_mae`) | `Mamba_MAE.pth` (MAE trên BraTS MRI) | zscore | 384 | `mixer.in_proj/out_proj` |
| `vmamba_3d` | VMamba-B, SS2D → SS3D | `vssm_base_VMamba.pth` ImageNet inflate | window+ImageNet | 1024 | `op.in_proj/out_proj` |
| `penet_3d` | PENet (ResNeXt-50 3D + SE) | `penet_best.pth.tar` | penet | 2048 | `encoders.3` |
| `ct_fm` / `ct_fm_features` | SegResEncoder / passthrough cache | CT-FM `model.safetensors` | remap [-1024,2048] | 512 | `layers.3/4.blocks` / (frozen only) |

File nguồn 3D ở `source/model/3D_model/<tên>.py`; run config ở `configs/runs/02_diagnosis/baselines/`.
Arm Mamba cần `mamba_ssm` (CUDA), `_thirdparty.py:require_mamba_ssm` hướng dẫn cài.

### 5.3 Inflate và weight

- `inflate_kernel` (`source/model/inflate.py`): kernel 2D `[O,I,kh,kw]` đặt vào mặt axial (x,y),
  lặp `kz` lần theo z và chia `kz` (I3D-style, volume hằng theo z cho đúng response 2D);
  `collapse_input_channels` gộp stem RGB → 1 kênh bằng **tổng**. `inflate_state_dict` copy khi
  shape khớp, inflate khi 4-D gặp 5-D, còn lại skip và báo.
- `load_state_with_report`: copy tensor khớp tên + shape, báo missing/unexpected; **raise nếu khớp
  < 50%**, không cho model "pretrained" mà phần lớn random. `model.pretrained.enabled` /
  `required` (false → scratch nhưng log rõ).

`python tools/baselines/prepare_weights.py --models vit_3d swin_3d` tải weight một lần trước khi train song song; `python tools/baselines/smoke.py --device cpu --size 32` kiểm tra wiring.

---

## 6. `BaselineClassifier` và head MLP / KAN

```text
volume → encoder.forward_features → global_embedding [B,F]
       → (tuỳ chọn) PooledFeatureStandardizer     # head.standardize_inputs
       → projection: Dropout → Linear(F, P) → LayerNorm(P)    P = head.projection_dim (64)
       → head (MLP | KAN): P → Σ output_dim các target → cắt thành logits theo target
```

Projection dùng chung cho cả hai head nên ablation head chỉ đổi đúng head. `fit_input_standardizer` fit z-score trên train split (all-reduce khi DDP), lưu thành buffer checkpoint, dùng cho encoder frozen. Output theo contract task model (diagnosis `{"logits": {target: [B,dim]}, ...}`; prognosis `{"logits": [B], "target_logits": ...}`) để dùng chung loss/trainer/evaluator.

Head: `MLPHead` (Linear(P, hidden) → GELU → Dropout → Linear(hidden, out); hidden 64, dropout 0.1) hoặc `KANHead` (pykan `KAN`, `third_party/repos/pykan`, width `[P, 16, out]`, grid 5, spline 3; luôn fp32). Preset ở `configs/components/baselines.yaml` (`#head_mlp`, `#head_kan`, `#train_end_to_end`, `#train_lora` LoRA r8/α16, `#train_frozen_features`).

---

## 7. PEFT (`source/components/peft/`)

`apply_peft(encoder, config["peft"])` → `peft_report`.

`full`: mọi tham số encoder trainable. `frozen`: đóng băng toàn encoder trừ `peft_trainable_modules`. `linear_probe`: **giống hệt** `frozen` trong code (chỉ khác tên trong report). `lora`: đóng băng encoder, `inject_lora` vào target, rồi mở lại `peft_trainable_modules`.

- Target LoRA: `encoder.lora_target_modules` (khai báo theo backbone) **ưu tiên** hơn
  `peft.target_modules` (generic).
- `peft_trainable_modules`: phần random-init mà checkpoint không phủ (attention pool slice-MIL);
  đóng băng nó thì head sẽ đứng sau một lớp random cố định.
- **Norm layer frozen:** dưới LoRA, BatchNorm/InstanceNorm bị đóng băng được đưa về `eval` (§2) nên
  running stats pretrained không bị batch nhỏ làm trôi.
- `LoRALinear`: `y = W x + (α/r)·B(A(dropout(x)))`, `A` kaiming, `B = 0` nên ban đầu bằng base; `LoRAConv`: `A` = conv rank-r cùng kernel/stride/padding, `B` = conv 1×1.
- Property `weight` trả **weight đã merge** `W + scale·BA`, vì code upstream đôi khi đọc thẳng `layer.weight` (Mamba fast path `in_proj.weight @ x`) mà không gọi layer.
- `inject_lora` bọc `Linear`/`Conv1d/2d/3d` có tên chứa token target, **bỏ qua conv grouped/depthwise** (ConvNeXt `conv_dw`, ResNeXt, Mamba `conv1d`); không khớp module nào → `ValueError`.

---

## 8. Forward pass có shape

Shape **đo trên CPU** với input cache thật 128³, B = batch.

### (a) Baseline 3D: `resnet18_3d`

```text
volume            [B, 1, 128, 128, 128]   RAS, [0,1] của [-1000,1000] HU
 └ IntensityAdapter(zscore)
 └ MonaiResNetCore            feature_map [B, 512, 4, 4, 4]
 └ mean(2,3,4)                global_embedding [B, 512]
 └ projection                 Dropout → Linear(512,64) → LayerNorm   [B, 64]
 └ MLPHead                    Linear(64,64) → GELU → Dropout → Linear(64,1)
 └ logits                     {"pe_present": [B, 1]}        (33,023,361 tham số, peft full)
```

### (b) 2.5D slice-MIL: `resnet18_25d`

```text
volume [B,1,128,128,128]
 └ IntensityAdapter(window [-250,450] → [0,1])
 └ uniform_slice_indices(128, 32); mỗi instance = slice (i-1,i,i+1) làm 3 kênh
     images       [B·32, 3, 128, 128] → bilinear [B·32, 3, 224, 224] → ImageNet norm
 └ Timm2DBackbone             maps [B·32, 512, 7, 7] → feature_map [B, 512, 7, 7, 32]
 └ mean(h,w)                  instances [B, 32, 512]
 └ GatedAttentionPool         global_embedding [B, 512]  (+ slice_attention [B, 32])
 └ projection [B,64] → head → {"pe_present": [B, 1]}      (11,345,154 tham số)
```
