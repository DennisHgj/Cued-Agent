"""Public package for the Cued-Agent inference pipeline."""

from __future__ import annotations

__all__ = ["CuedAgentInference", "__version__"]
__version__ = "1.1.0"


def __getattr__(name: str):
    """Keep the package importable without loading optional ML dependencies."""
    if name == "CuedAgentInference":
        from .pipeline import CuedAgentInference

        return CuedAgentInference
    raise AttributeError(name)
