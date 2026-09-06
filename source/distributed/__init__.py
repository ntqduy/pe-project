"""Distributed helpers are imported lazily so CPU preflight works without PyTorch."""

__all__ = ["DistributedContext", "initialize_distributed", "gather_prediction_rows"]


def __getattr__(name):
    if name in {"DistributedContext", "initialize_distributed"}:
        from .setup import DistributedContext, initialize_distributed

        return {"DistributedContext": DistributedContext, "initialize_distributed": initialize_distributed}[name]
    if name == "gather_prediction_rows":
        from .gather import gather_prediction_rows

        return gather_prediction_rows
    raise AttributeError(name)
