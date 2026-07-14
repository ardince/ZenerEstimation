"""
Forecasting framework.

This package contains the forecasting pipeline and
model registry used by all forecasting algorithms.
"""

from .pipeline import ForecastPipeline
from .models import ModelRegistry
from .base import BaseForecastModel
from .result import ForecastResult

__all__ = [
    "ForecastPipeline",
    "ModelRegistry",
    "BaseForecastModel",
    "ForecastResult",
]