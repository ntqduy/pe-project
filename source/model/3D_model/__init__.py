"""Volumetric baselines, one file per architecture family.

resnet.py    ResNet-18 / ResNet-50 3D (MONAI), MedicalNet 23-dataset weights
densenet.py  DenseNet-121 3D (MONAI), ImageNet weights inflated 2D -> 3D
convnext.py  ConvNeXt-T 3D (native), ImageNet weights inflated 2D -> 3D
vit.py       ViT-S/16 3D (timm blocks + 3D patch embedding), ImageNet weights inflated
swin.py      Swin 3D (MONAI SwinTransformer), Swin-UNETR self-supervised CT weights
nnmamba.py   nnMamba4cls (third_party/repos/nnMamba), no public weights
mamba_mae.py 3D Vision Mamba from Mamba-MAE (third_party/repos/mamba_mae), MAE weights
vmamba.py    VMamba-B 3D (native SS3D on VMamba primitives), ImageNet weights inflated
penet.py     PENet (third_party/repos/penet), released PE weights, fine-tuned
CT-FM LoRA / frozen reuse source/components/encoders/image/ct_fm.py.
"""
