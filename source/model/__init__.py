"""Baseline model zoo: 2D / 2.5D slice-MIL and 3D image encoders plus MLP / KAN heads.

Layout
------
``2D_model/``   per-slice ImageNet backbones (timm) and a MIL wrapper that turns them into
                whole-volume encoders (2D = one slice per instance, 2.5D = three channels).
``3D_model/``   volumetric encoders (ResNet, DenseNet, ConvNeXt, ViT, Swin, nnMamba,
                Mamba-MAE, VMamba, PENet); CT-FM reuses ``source/components/encoders``.
``head/``       the classification heads compared in the head ablation (MLP, KAN).

Every encoder here is a ``BaseImageEncoder`` (``source/components/encoders/image/base.py``)
so the existing trainer, evaluator, Grad-CAM preview and checkpoint lineage work unchanged.
Each one is registered as a ``model.backbone`` name in ``source/model/registry.py``; the
contracts (feature_dim, pretrained source, intensity mapping) live in
``configs/components/backbones.yaml``.

``2D_model`` / ``3D_model`` are not valid Python identifiers, so they are only ever imported
by string through ``importlib`` (see ``registry.py``); relative imports inside them work.
"""
