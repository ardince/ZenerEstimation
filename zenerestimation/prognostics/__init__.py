"""
Prognostics package.

Provides Remaining Useful Life (RUL) estimation,
threshold analysis and future prognostic algorithms.
"""

from .result import PrognosticResult
from .threshold import ThresholdEstimator

__all__ = [
    "PrognosticResult",
    "ThresholdEstimator",
]