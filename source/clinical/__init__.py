from .pesi import PESIResult, compute_pesi, compute_spesi

__all__ = [
    "ClinicalEncoder",
    "ClinicalPreprocessor",
    "PESIResult",
    "compute_pesi",
    "compute_spesi",
]


def __getattr__(name: str):
    """Keep the pure-Python PESI calculator usable by the data-preprocessing stage.

    Importing ``source.clinical.pesi`` first imports this package.  Eagerly importing
    the neural encoder here would make a metadata/PESI audit require PyTorch even though
    it neither trains nor loads a model.  Training imports retain the same public API,
    while the heavy modules load only when a caller actually requests them.
    """
    if name == "ClinicalEncoder":
        from .encoder import ClinicalEncoder

        return ClinicalEncoder
    if name == "ClinicalPreprocessor":
        from .preprocessing import ClinicalPreprocessor

        return ClinicalPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
