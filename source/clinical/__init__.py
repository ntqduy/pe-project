from .spesi import SPESI_COMPONENTS, SPESI_FEATURE_COLUMNS, SpesiResult, compute_spesi

__all__ = [
    "ClinicalEncoder",
    "ClinicalPreprocessor",
    "SPESI_COMPONENTS",
    "SPESI_FEATURE_COLUMNS",
    "SpesiResult",
    "compute_spesi",
]


def __getattr__(name: str):
    """Load the heavy clinical modules only when a caller actually requests them.

    A metadata-only stage-0 pass imports this package but neither trains nor loads a
    model, so eagerly importing the neural encoder here would make it require PyTorch.
    """
    if name == "ClinicalEncoder":
        from .encoder import ClinicalEncoder

        return ClinicalEncoder
    if name == "ClinicalPreprocessor":
        from .preprocessing import ClinicalPreprocessor

        return ClinicalPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
