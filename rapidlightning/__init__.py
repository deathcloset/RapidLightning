"""
Rapid Lightning — nanosecond-class tabular classification.

A head-aware genetic-programming ensemble that compiles to a ~15 KB,
dependency-free C `.so` predicting in tens of nanoseconds.

    from rapidlightning import RapidLightningClassifier

See the demos/ folder for runnable showcases.
"""
from .classifier import RapidLightningClassifier
from .forge import CompiledPredictor

__all__ = ["RapidLightningClassifier", "CompiledPredictor"]
__version__ = "0.1.0"
