# 02. Models

> **Lưu ý (2026-10-03):** các experiment anatomy-aware / Soft-MoE / late-logit / global / matrix, toàn bộ `03_prognosis` (modality, EHR ablation), `04_anatomy_analysis` (counterfactual, architecture) và external Turkey test đã được gỡ khỏi `configs/` và `configs/experiments.yaml`; phần nhắc tới chúng dưới đây chỉ còn giá trị lịch sử (config cũ: `git show 41e4c8c:configs/runs/...`). Protocol đang dùng: baseline zoo exp01-exp04 (`scripts/diagnosis/baselines/README.md`), CT-FM frozen, zero-shot PENet/RADAR và pipeline dữ liệu `00_data`.

**Đọc khi:** cần biết một config biến thành `nn.Module` thế nào; muốn thêm/đổi backbone; muốn hiểu
forward pass (tensor shape) của baseline 3D, 2.5D slice-MIL và model anatomy-aware; muốn biết PEFT
(full / frozen / LoRA) thực sự đóng băng cái gì.

**Code chính:** `source/engine/factory.py`, `source/components/encoders/image/`, `source/model/`
(`base.py`, `classifier.py`, `registry.py`, `inflate.py`, `weights.py`, `2D_model/`, `3D_model/`,
`head/`), `source/components/{peft,anatomy.py,roi/pooling.py,adapters,fusion}`,
`source/tasks/{diagnosis,prognosis}/model.py`,
`configs/components/{backbones,encoders,baselines,fusions,anatomy}.yaml`.

Dữ liệu và ROI mask ở [01_data_pipeline.md](01_data_pipeline.md); loss, trainer, evaluation ở
[03_training_evaluation.md](03_training_evaluation.md); các arm thí nghiệm (baseline grid,
anatomy, counterfactual) ở [04_experiments.md](04_experiments.md).

---

## 1. Lắp model: `build_task_model`

`source/engine/factory.py:build_task_model(config) -> (model, peft_report)` là điểm vào duy nhất
(train, evaluate, counterfactual, Grad-CAM đều gọi nó).

```text
config  (experiment.stage ∈ {diagnosis, prognosis}; ablation/counterfactual dùng task.base_stage)
  └─ task.architecture
        ├─ "baseline_classifier" -> build_image_encoder -> apply_peft
        │                           -> BaselineClassifier(encoder, targets, head)
        ├─ stage=diagnosis -> build_image_encoder -> apply_peft
        │                     -> DiagnosisModel(encoder, targets, regions, organ_adapter, fusion, ...)
        └─ stage=prognosis -> image encoder (nếu "image" ∈ task.modalities) -> apply_peft
                              + EHREncoder ("ehr") + SpesiEncoder ("spesi") -> PrognosisModel(...)
```

- **Hai họ model**: *baseline zoo* (`baseline_classifier`, chỉ dùng `global_embedding`, không có
  ROI) và *anatomy-aware task model* (`DiagnosisModel`/`PrognosisModel`, pool feature map theo mask
  heart/PA/lung rồi fuse).
- PEFT chỉ áp lên **image encoder**; projection, adapter, fusion, head luôn trainable.
- Baseline từ chối prognosis đa modality (`task.modalities` phải là `[image]`). Với arm CT-FM, cờ
  `--scratch` dịch thành `model.load_pretrained=false`; với `cached_features` thì scratch bị cấm
  (feature đã trích bằng weight pretrained).
- Chọn backbone: `--set model.backbone=resnet18_3d`; `resolve_backbone` (`source/utils/config.py`)
  copy entry tương ứng trong `configs/components/backbones.yaml#registry` vào `model`. Đổi backbone
  khác preset sẽ stamp `__bb_<backbone>` vào run id.

---

## 2. Hợp đồng encoder: `BaseImageEncoder`

File `source/components/encoders/image/base.py`. Mọi encoder trả `ImageFeatures`:

`ImageFeatures`: `feature_map [B, C_map, d, h, w]` (luôn 5-D, theo trục input RAS), `global_embedding [B, feature_dim]` (luôn 2-D), `pyramid` (level trung gian), `metadata` (`{"backbone", "pooling"}`; `pooling` quyết định global branch ở §8).

`forward(volume)` trả `global_embedding`. `feature_map_dim` (số kênh `feature_map`, cái mà ROI
pooling nhìn thấy) mặc định bằng `feature_dim`; chỉ khác khi embedding không phải mean của map
(nnMamba: embedding 448 = concat mean 3 level, map 256).

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
   `-` → `_`). Đăng ký sẵn `ct_fm`, `ct_fm_features`, `penet_style` và toàn bộ baseline zoo. Tên lạ →
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

- `CTFMInputContractEncoder` là **subclass** của `SegResEncoder` nên state-dict key giữ nguyên → strict load đủ 161 tensor. Feature map trả về theo RAS nên mask vùng và counterfactual khớp. Spacing **không** được chuyển (cache 1.5 mm khác 3×1×1 mm của CT-FM). Input 128³ cho `feature_map [B,512,8,8,8]`.
- LoRA: `SegResEncoder` không có `nn.Linear`, nên `lora_target_modules: [layers.3.blocks, layers.4.blocks]` (conv hai stage sâu nhất).

### 4.2 CT-FM cached features (`backbone: ct_fm_features`)

Nhét cả ngực vào một patch 24×128×128 cho voxel ~12×1.9×2.6 mm là scale CT-FM chưa từng thấy, nên
`tools/data/build_ctfm_cache.py` trích feature **một lần/study** theo contract upstream (xem
[01_data_pipeline.md](01_data_pipeline.md) §1.6).

- Tensor cache `[B, 513, d, h, w]` = 512 kênh feature + 1 kênh "valid" (tỉ lệ ô trong body box).
  `CTFMFeaturePassthrough` chỉ tách hai phần.
- `global_embedding` = **mean có trọng số body** (loại ô padding không khí), `metadata.pooling =
  "body_weighted_mean"`.
- `CachedCTFMEncoder` không có tham số; chỉ dùng frozen, **không** làm counterfactual ảnh
  (`tools/_common.py` chặn khi `model.cached_features`). Arm `ctfm_frozen_3d` đọc bản pooled
  `[513,1,1,1]`.

### 4.3 PENet

`penet_style` (`encoders/image/penet.py`, `diag.global.penet_style`): CNN residual 3D tự viết, **random init**, `feature_dim` 256, không phải tái hiện PENet gốc. `penet_3d` (`3D_model/penet.py`): PENet thật (`third_party/repos/penet`, weight `penet_best.pth.tar` load strict + SHA-256), RAS → layout DICOM của PENet, intensity `penet`, giữ stochastic depth nên `encoder.ddp_find_unused_parameters = True`. Zero-shot (`penet_zeroshot.py`, `diag.zeroshot.penet`) đọc NIfTI gốc, không train.

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
  `slice_attention`, `slice_depth` (ROI pooling đọc mask đúng trên các slice này).
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

## 8. Đường anatomy-aware (`source/components/anatomy.py`)

Ý tưởng: **encode cả CTPA một lần**, rồi pool cùng một feature map thành vector global và vector
theo vùng. `z_heart`, `z_pa`, `z_lung` *không* chỉ chứa thông tin của cơ quan: receptive field của
encoder nhìn cả volume.

`pool_anatomy_features(encoder, roi, volume, masks)`:

1. `image = encoder.forward_features(volume)`.
2. `z_global` theo `metadata.pooling`: `body_weighted_mean` (CT-FM cached) → dùng `global_embedding`
   (plain mean sẽ trộn vector "air" theo kích thước cơ thể); bắt đầu bằng `slice_mil_` → dùng
   aggregate MIL (đồng thời để attention pool nhận gradient, tránh lỗi DDP unused params); còn lại →
   mean không gian của `feature_map`.
3. `ROIFeatureExtractor(regions)` gọi `mask_guided_pool` cho từng vùng (thiếu mask → `KeyError`).
4. `available = {"global": True, <region>: present}`.

**ROI coverage pooling (`components/roi/pooling.py`).** `resize_mask` co mask về lưới feature bằng
`adaptive_avg_pool3d` nên **trọng số = tỉ lệ ô nằm trong vùng**; nearest sẽ chỉ đọc một voxel mỗi ô
và làm mất vùng nhỏ như PA. `mask_guided_pool`: `pooled = Σ f·w / Σ w`; `present` đọc trên **mask
gốc** (`amax > ε`), không trên tổng trọng số. Vùng vắng → vector 0, `present = False`.

**Slice-MIL.** `select_mask_slices` lấy mask trung bình đúng trên các slice (hoặc bộ 3 slice) mà mỗi
instance đã encode, tạo mặt trọng số `[B,1,x,y,N]`, nên pooling theo vùng khớp với feature map `[B,C,h,w,N]`.

**Organ adapter (`components/adapters/`).** `OrganAdapterBank(input_dim=feature_map_dim, output_dim=expert_dim, regions)`: một adapter **độc lập** cho mỗi branch (`global` + từng region). Loại: `bottleneck_mlp` (mặc định: Linear → LayerNorm → GELU → Dropout → Linear), `residual` (skip + bottleneck, `up` init 0), `lora`. `standardize_inputs: true` → z-score từng branch fit trên train; output nhân với `present` nên branch vắng thành vector 0. Hidden width = `task.hidden_dim` (256). Nguồn mask: `configs/components/anatomy.yaml#anatomy_masks` (`heart: ROI2`, `pa: ROI4`, `lung: ROI6`).

---

## 9. Fusion (`source/components/fusion/`)

`resolve_fusion_type` chuẩn hoá `fusion.type` (fallback `task.architecture`; preset ở `configs/components/fusions.yaml`):

| Canonical (alias) | Mức | Cơ chế | Branch vắng |
|---|---|---|---|
| `concat_mlp` (`concat`) | feature | concat → Linear → LayerNorm → GELU → Dropout → Linear | nhân 0 trước khi concat (vẫn chiếm chỗ) |
| `soft_moe` (`moe`) | feature | router MLP trên concat → logits/`temperature` → softmax → tổng có trọng số các expert | logit router = `-inf` rồi **renormalize**; mọi branch vắng → lỗi |
| `late_logit` (`late`) | logit | mỗi branch một head; trung bình logit có trọng số (`learned: true` → softmax `log_weights`) | bỏ khỏi trung bình và renormalize |

---

## 10. Task model

**`DiagnosisModel` (`source/tasks/diagnosis/model.py`).** Shared encoder → organ adapters →
prediction head → fusion. Chỉ dựng **một** trong hai đường: feature fusion (`self.fusion` +
`self.heads = DiagnosisHeads`, một `nn.Linear` mỗi target) hoặc late-logit
(`branch_heads` + `logit_fusion`). Tên module `image_encoder, roi, organ_adapters, fusion, heads`
được giữ ổn định vì `lineage.transfer_modules` của counterfactual liệt kê đúng các tên này.

- `DEFAULT_TARGETS`: `pe_present` + 13 target phụ. Auxiliary organ heads (`task.auxiliary_targets`) gắn vào branch adapted của **đúng cơ quan** (heart/pa/lung), mỗi target thuộc một cơ quan (`organ_targets.py`), nguồn `native | silver | expert_reviewed`.
- Output: `logits`, `auxiliary_logits`, `routing` (`None` với concat), `features`, `branch_features`, `roi_present`. Loss (`losses.py`): BCE-with-logits / CE chỉ trên nhãn valid, nhân `task.loss_weights`; `organ_auxiliary_loss` masked theo nguồn nhãn; chi tiết ở [03_training_evaluation.md](03_training_evaluation.md).

**`PrognosisModel` (`source/tasks/prognosis/model.py`).** Cùng stack ảnh như diagnosis (kiến trúc
không là confound), thêm branch `ehr` (`EHREncoder`) và `spesi` (`SpesiEncoder`) **chỉ ở bước
fusion**, mỗi branch lâm sàng có adapter riêng về cùng width. `task.modalities` chọn branch; branch
thiếu dữ liệu bị mask; image encoder có thể `None` (EHR-only).

---

## 11. Forward pass có shape

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

### (c) Anatomy-aware diagnosis: CT-FM + ROI branch + concat fusion

```text
volume [B,1,128,128,128]     masks heart(ROI2) / pa(ROI4) / lung(ROI6) [B,1,128,128,128]
 └ CT-FM (remap HU, RAS→SPL, SegResEncoder, SPL→RAS)   feature_map [B, 512, 8, 8, 8]
     ├─ global: adaptive_avg_pool3d ─────────────────────► z_global [B,512]   present=1
     ├─ heart : resize_mask (coverage) → w [B,1,8,8,8]; Σ f·w / Σ w ─► z_heart [B,512] present_heart [B]
     ├─ pa    : ...                                      ─► z_pa   [B,512]
     └─ lung  : ...                                      ─► z_lung [B,512]
 └ OrganAdapterBank (bottleneck_mlp, độc lập mỗi branch): 512 → 256 → 128, × present
     adapted_{global,heart,pa,lung}  4 × [B,128]
 └ ConcatMLPFusion: concat [B,512] → Linear(512,128) → LN → GELU → Dropout → Linear(128,128)
     fused [B,128]                  routing = None
 └ DiagnosisHeads: Linear(128, dim) mỗi target → logits {"pe_present": [B,1], ...}

Biến thể soft_moe: router concat [B,512] → MLP → Linear(·,4) → /T → mask branch vắng → softmax;
   fused = Σ_k weight[B,k]·adapted_k [B,128]; routing [B,4]
```

---