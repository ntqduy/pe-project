"""Reusable neural-network components shared by all scientific tasks."""

from .adapters import OrganAdapterBank
from .targets import TargetSpec, normalize_target_specs

__all__ = ["OrganAdapterBank", "TargetSpec", "normalize_target_specs"]
