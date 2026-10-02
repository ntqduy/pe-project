"""Classification heads compared in the head ablation: MLP and KAN.

Both sit on the same shared projection (``source/model/classifier.py``), so the only thing
that differs between an ``mlp`` and a ``kan`` run is the head itself.
"""
from .factory import HEAD_TYPES, build_head

__all__ = ["HEAD_TYPES", "build_head"]
