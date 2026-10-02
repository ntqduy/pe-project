"""2-D and 2.5-D baselines: ImageNet backbones inside a slice-MIL volume encoder.

``timm2d.py``   timm backbone -> per-slice feature map [N, C, h, w]
``mil.py``      slices of the volume -> backbone -> MIL pooling -> whole-volume embedding
``resnet.py`` / ``convnext.py`` / ``vit.py`` / ``swin.py``   one builder per architecture
"""
