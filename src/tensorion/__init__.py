"""Tensorion: a drop-in replacement for Muon that also handles tensor layers."""

from . import orthogonalize, unfolding
from .optimizer import Tensorion

__version__ = "0.1.0"

__all__ = ["Tensorion", "orthogonalize", "unfolding", "__version__"]
